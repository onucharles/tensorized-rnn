from comet_logger import CometLogger

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path

from utils import data_generator
from mnist_classifier import MNIST_Classifier


parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--ttlstm', action='store_true',
                    help='use tensorized LSTM (default: False)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--n_layers', type=int, default=2,
                    help='# of layers (default: 2)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--hidden_size', type=int, default=32,
                    help='number of hidden units per layer (default: 32)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: False)')
parser.add_argument('--ncores', type=int, default=3,
                    help='number of TT cores (default: 3)')
parser.add_argument('--ttrank', type=int, default=2,
                    help='TT rank (default: 2)')
parser.add_argument('--enable_logging', action='store_true',
                    help='Log metrics to Comet and save model to disk (default: False)')
parser.add_argument('--models_dir', type=str, help='Path to saved model files.')

args = parser.parse_args()

# create comet logger.
logger = CometLogger(not args.enable_logging, is_existing=False, prev_exp_key=None)
run_id = logger.get_experiment_key()
logger.log_params(vars(args))

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    # set default cuda device.
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    # warn if not using cuda and gpu is available.
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = './data/mnist'
batch_size = args.batch_size
n_classes = 10
input_channels = 28
seq_length = int(784 / input_channels)
epochs = args.epochs
steps = 0

print(args)
train_loader, test_loader = data_generator(root, batch_size)

permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
model = MNIST_Classifier(input_channels, n_classes, args.hidden_size, args.n_layers, device,
                         tt_lstm=args.ttlstm, n_cores=args.ncores, 
                         tt_rank=args.ttrank)

if args.cuda:
    model.cuda()
    permute = permute.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

def train(ep):
    global steps
    train_loss = 0
    train_correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, seq_length, input_channels)
        if args.permute:
            data = data[:, :, permute]
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        train_loss += loss
        steps += 1
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            avg_train_loss = train_loss.item() / args.log_interval
            avg_train_acc = 100. * train_correct.item() / (args.log_interval * batch_size)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: ({:.2f}%)\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), avg_train_loss, avg_train_acc, steps))
            logger.log_metrics({"loss": avg_train_loss, "accuracy": avg_train_acc},
                               prefix="train", step=steps)
            train_loss = 0
            train_correct = 0

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, seq_length, input_channels)
            if args.permute:
                data = data[:, :, permute]
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_acc))
        logger.log_metrics({"accuracy": test_acc, "loss": test_loss}, prefix="test", step=steps)
        return test_loss


if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
