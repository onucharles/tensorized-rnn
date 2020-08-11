from time import time
from comet_logger import CometLogger

import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torch.autograd import Variable

from utils import data_generator, count_model_params
from mnist_classifier import MNIST_Classifier


parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--tt', action='store_true',
                    help='use tensorized RNN model (default: False)')
parser.add_argument('--gru', action='store_true',
                    help='use GRU instead of LSTM (default: False)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--n_layers', type=int, default=1,
                    help='# of layers (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='initial learning rate (default: 1e-2)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--lr_scheduler', action='store_false',
                    help='Whether to use piecewise-constant LR scheduler '
                         '(default: True)')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units per layer (default: 256)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: False)')
parser.add_argument('--ncores', type=int, default=2,
                    help='number of TT cores (default: 2)')
parser.add_argument('--ttrank', type=int, default=2,
                    help='TT rank (default: 2)')
parser.add_argument('--enable_logging', action='store_true',
                    help='Log metrics to Comet and save model to disk (default: False)')
parser.add_argument('--log_grads', action='store_true',
                    help='Whether to log gradients and activations (default: False)')
parser.add_argument("--gpu_no", type=int, default=0, help =\
                "The index of GPU to use if multiple are available. If none, CPU will be used.")
# parser.add_argument('--models_dir', type=str, help='Path to saved model files.')

args = parser.parse_args()
if not args.tt:
    args.ncores = 1
    args.ttrank = 1

# create comet logger.
logger = CometLogger(not args.enable_logging)
run_id = logger.get_experiment_key()
mod_name = 'gru' if args.gru else 'lstm'
logger.log_params(vars(args))
name = (f"{mod_name}-{'tt' if args.tt else 'no-tt'}-n{args.n_layers}"
        f"-h{args.hidden_size}-ncores{args.ncores}-rank{args.ttrank}")
logger.set_name(name)

# saved model path
model_dir = Path("saved_models") / run_id
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "model.pt"

# fix seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# Set cuda device
if torch.cuda.is_available():
    # set default cuda device.
    device = torch.device('cuda:{}'.format(args.gpu_no))
    torch.cuda.set_device(device)

    # warn if not using cuda and gpu is available.
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    device = torch.device('cpu')

root = './data/mnist'
batch_size = args.batch_size
n_classes = 10
epochs = args.epochs
steps = 0
if args.permute:    # each pixel is input for a time step.
    input_channels = 1
else:               # each row(of pixels) is input for a time step.
    input_channels = 28
seq_length = int(784 / input_channels)

print(args)
train_loader, val_loader, test_loader = data_generator(root, batch_size)

model = MNIST_Classifier(input_channels, n_classes, args.hidden_size, args.n_layers, device,
                         tt=args.tt, gru=args.gru, n_cores=args.ncores, 
                         tt_rank=args.ttrank, log_grads=args.log_grads)
n_trainable, n_nontrainable = count_model_params(model)
print("Model instantiated. Trainable params: {}, Non-trainable params: {}. Total: {}"
      .format(n_trainable, n_nontrainable, n_trainable + n_nontrainable))

# Setup activation and gradient logging
if args.log_grads:
    from tensorized_rnn.grad_tools import ActivGradLogger as AGL

permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
if args.cuda:
    model.cuda()
    permute = permute.cuda()

# Set learning rate, optimizer, scheduler
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
if args.lr_scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

# print(f"input channels: {input_channels}; seq_length: {seq_length}; cuda: {args.cuda}")
# exit()
def train(ep):
    global steps
    train_loss = 0
    train_correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, seq_length, input_channels)
        if args.permute:
            data = data[:, permute, :]
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if args.log_grads: AGL.end_minibatch()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        train_loss += loss
        # print(f"step: {steps}; loss: {loss}")
        steps += 1
        if batch_idx > 0 and (batch_idx + 1) % args.log_interval == 0:
            avg_train_loss = train_loss.item() / args.log_interval
            avg_train_acc = 100. * train_correct.item() / (args.log_interval * batch_size)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: ({:.2f}%)\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), avg_train_loss, avg_train_acc, steps))
            logger.log_metrics({"loss": avg_train_loss, "accuracy": avg_train_acc},
                               prefix="train", step=steps)
            train_loss = 0
            train_correct = 0
    if args.log_grads: AGL.end_epoch()



best_test_acc = 0.0
def test(test_model, loader, val_or_test="val"):
    test_model.eval()
    test_loss = 0
    correct = 0
    global best_test_acc
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, seq_length, input_channels)
            if args.permute:
                data = data[:, permute, :]
            data, target = Variable(data), Variable(target)
            output = test_model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            if args.log_grads: AGL.del_record()

        test_loss /= len(loader.dataset)
        test_acc = 100. * correct / len(loader.dataset)
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            val_or_test, test_loss, correct, len(loader.dataset), test_acc))
        logger.log_metrics({"accuracy": test_acc, "loss": test_loss},
                prefix=val_or_test, step=steps)

        # if validation track best accuracy and save best model.
        if val_or_test == "val" and not logger.disabled:
            if test_acc - best_test_acc > 1e-5:
                best_test_acc = test_acc
                torch.save({"model_state": test_model.state_dict(),
                    "step": steps, "val_acc": test_acc}, model_path)
                logger.log_metric("best_val_acc", best_test_acc, step=steps)
        return test_loss


if __name__ == "__main__":
    import os.path
    from tensorized_rnn.grad_tools import ActivGradLogger as AGL
    start = time()

    for epoch in range(1, epochs+1):
        train(epoch)
        test(model, val_loader, "val")
        if args.lr_scheduler: scheduler.step()

        logger.save_act_grads(AGL.get_logs())
        assert os.path.isfile(f"./.{name}.record")
        print(f"Runtime: {time() - start:.0f} sec\n")
        # if epoch % 10 == 0:
            # lr /= 10
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr
    
    # if logging is enabled, then run best model on test set.
    if not logger.disabled:
        checkpoint = torch.load(model_path)
        print("Loading best model from step {} with val_acc: {}..."
                .format(checkpoint["step"], checkpoint["val_acc"]))
        model.load_state_dict(checkpoint["model_state"])
        test(model, test_loader, "test")
