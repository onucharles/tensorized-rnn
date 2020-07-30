import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

from context import tensorized_rnn
from tensorized_rnn.grad_tools import get_data

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--tt', action='store_true',
                    help='use tensorized RNN model (default: False)')
parser.add_argument('--gru', action='store_true',
                    help='use GRU instead of LSTM (default: False)')
parser.add_argument('--n_layers', type=int, default=1,
                    help='# of layers (default: 1)')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units per layer (default: 256)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: False)')
parser.add_argument('--ncores', type=int, default=4,
                    help='number of TT cores (default: 2)')
parser.add_argument('--ttrank', type=int, default=5,
                    help='TT rank (default: 2)')
parser.add_argument('--locations', type=str, default='0',
                    help='locations at which quantities are plotted')
parser.add_argument('--quantity', type=str, default='act',
                    help='The quantity being plotted')

args = parser.parse_args()
if not args.tt:
    args.ncores = 1
    args.ttrank = 1
mod_name = 'gru' if args.gru else 'lstm'

# Name of save file
name = (f".{mod_name}-{'tt' if args.tt else 'no-tt'}-n{args.n_layers}"
        f"-h{args.hidden_size}-ncores{args.ncores}-rank{args.ttrank}.record")
fig_name = f"figures/{name[1:-7]}-{args.quantity}.png"

# Get the data for this plot 
quantities = {'act': 'Mean Activation', 'log_act': 'Mean Log Activation', 
              'grad': 'Mean Gradient', 'log_grad': 'Mean Log Gradient'}
assert args.quantity in quantities
quant = quantities[args.quantity]
data_mat = get_data(name, args.quantity)

# Pull out the time series of interest
locations = [int(s) for s in args.locations.split()]
time_series = [np.array(data_mat[:, n-1]) for n in locations]
num_epochs, _ = data_mat.shape

# Set up the plot
fig, ax = plt.subplots()
x_axis = np.arange(1, num_epochs+1)
ax.set(xlabel='Epoch', ylabel=quant)
       # title='')

# Plot the time series
for ts, loc in zip(time_series, locations):
    ax.plot(x_axis, ts, label=f't = {loc+1}')

ax.legend()
fig.savefig(fig_name, format='png')
