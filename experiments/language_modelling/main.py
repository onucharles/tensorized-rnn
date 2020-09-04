# coding: utf-8
import argparse
import time
import math
import os
from comet_logger import CometLogger    # Must be imported before torch
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
from functools import partial

import data
import model

# Better printing in cluster environment
print = partial(print, flush=True)

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=5,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--train_frac', type=float, default=0.2,
                    help='Fraction of training data to use.')
parser.add_argument('--full_test', action='store_false',
                    help='use the full test set (default: True)')
parser.add_argument('--tt', action='store_true',
                    help='use tensorized RNN model (default: False)')
parser.add_argument('--naive_tt', action='store_true',
                    help='use naive tensorized RNN model (default: False).')
parser.add_argument('--ncores', type=int, default=2,
                    help='number of TT cores (default: 2)')
parser.add_argument('--ttrank', type=int, default=2,
                    help='TT rank (default: 2)')
parser.add_argument('--voc_pad', type=int, default=22,
                    help='amount of padding to vocabulary size.')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--patience', type=int,  default=4,
                    help='number of non-improving epochs before dropping lr')
parser.add_argument('--lr_drop', type=float,  default=10.,
                    help='amount to drop lr by')
parser.add_argument('--leadin', type=int,  default=5,
                    help='initial epochs before lr scheduler begins')
parser.add_argument('--tt_embedding', action='store_true',
                    help='Tensor train word embeddings (default: False)')
parser.add_argument('--enable_logging', action='store_true',
                    help='Log metrics to Comet (default: False)')

args = parser.parse_args()
print(args)

assert 0.0 < args.train_frac <= 1.0, "arg 'train_frac' must be betweetn (0.0  1.0]"
assert not (args.naive_tt and not args.tt)
if args.dry_run: args.enable_logging = False

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    args.cuda = False

device = torch.device("cuda" if args.cuda else "cpu")

# Set comet.ml logging
logger = CometLogger(not args.enable_logging)
logger.log_params(vars(args))
name = (f"{args.model.lower()}-{'nv' if args.naive_tt else ''}"
        f"{'tt-' if args.tt else ''}h{args.nhid}-n{args.nlayers}" + 
        (f"-ncores{args.ncores}-rank{args.ttrank}" if args.tt else '') + 
        f"-{int(100*args.train_frac)}%-{args.optimizer}")
logger.set_name(name)

# Use name of specific configuration to set model path
args.save = f"best_models/{name}.pt"
if not os.path.isfile(args.save): 
    print(f"Overwriting previous model file'{args.save}'")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data, args.train_frac, args.voc_pad, 
                     full_test=args.full_test)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, device,
                       args.dropout, args.tied, tt=args.tt, n_cores=args.ncores,
                       tt_rank=args.ttrank, naive_tt=args.naive_tt, 
                       tt_embedding=args.tt_embedding).to(device)
print(f"Number of parameters in model: {model.param_count()}")
criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()

    return total_loss / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    num_steps = 0
    total_loss = 0.
    running_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)

    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        # Randomize the length so we don't have same batches every epoch
        seq_len = args.bptt if np.random.random() < 0.95 else args.bptt // 2
        data, targets = get_batch(train_data, i, seq_len)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        #model.zero_grad()
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        #for p in model.parameters():
        #    p.data.add_(p.grad, alpha=-lr)
        optimizer.step()

        num_steps += 1
        total_loss += loss.item()
        running_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = running_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:1.0e} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            running_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

        batch += 1
        i += seq_len

    train_loss = total_loss / num_steps
    train_ppl = math.exp(train_loss) if train_loss < 15 else float('nan')
    logger.log_metric("train_loss", train_loss, epoch=epoch)
    logger.log_metric("train_ppl", train_ppl, epoch=epoch)


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
bad_epochs = 0
best_val_loss = 1e9

# At any point you can hit Ctrl + C to break out of training early.
try:
    # Initiialize the optimizer
    params = list(model.parameters())
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        val_ppl = math.exp(val_loss) if val_loss < 15 else float('nan')
        epoch_time = time.time() - epoch_start_time
        logger.log_metrics({"val_loss": val_loss, "val_ppl": math.exp(val_loss), 
                            "cur_lr": lr, "epoch_time": epoch_time}, epoch=epoch)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, epoch_time, val_loss, val_ppl))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            bad_epochs = 0
        else:
            # lr /= 4
            # Anneal the learning rate if we're out of patience
            if epoch > args.leadin: bad_epochs += 1
            if bad_epochs > args.patience:
                bad_epochs = 0
                lr /= args.lr_drop
                optimizer.param_groups[0]['lr'] = lr
            if lr < 1e-4:   # Too small for effective training
                break

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    # if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
    #     model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
test_ppl = math.exp(test_loss)
logger.log_metrics({"test_loss": test_loss, "test_ppl": test_ppl}, epoch=epoch)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, test_ppl))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
