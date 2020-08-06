"""
Benchmarking runtime of digit classification models.
"""

import torch
import argparse
from time import time
import numpy as np
from mnist_classifier import MNIST_Classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tt", action='store_true')
    parser.add_argument("--ncores", type=int, default=1)
    parser.add_argument("--ttrank", type=int, default=1)
    parser.add_argument("--gru", action='store_true')
    parser.add_argument("--permute", action='store_true')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument("-n", "--nruns", type=int, default=100)
    args = parser.parse_args()
    print(args)

    if args.permute:  # each pixel is input for a time step.
        input_channels = 1
    else:  # each row(of pixels) is input for a time step.
        input_channels = 28
    n_classes = 10
    seq_length = int(784 / input_channels)

    # instantiate model
    device = torch.device('cuda')
    model = MNIST_Classifier(input_channels, n_classes, args.hidden_size, args.n_layers, device,
                             tt=args.tt, gru=args.gru, n_cores=args.ncores,
                             tt_rank=args.ttrank).to(device)

    # create random batch of data (using appropriate sizes)
    random_batch = np.random.rand(args.batch_size, seq_length, input_channels).astype('float32')
    random_batch = torch.from_numpy(random_batch).to(device)
    print("Benchmarking with input of shape (batch_size, seq_length, input_channels): ",
          random_batch.shape)

    # run and time forward pass x times.
    model.eval()
    all_durations = []
    out = model(random_batch)   # do one time to wake gpu.
    with torch.no_grad():
        for i in np.arange(args.nruns):
            # time forward pass
            start = time()
            out = model(random_batch)
            duration = time() - start

            # save duration
            print(f"Run: {i} \t Duration: {duration}", )
            all_durations.append(duration)

    # print mean and std of durations.
    all_durations = np.array(all_durations)
    mean_time = np.mean(all_durations)
    std_time = np.std(all_durations)
    print(f"mean time: {mean_time} \t std time: {std_time}")
