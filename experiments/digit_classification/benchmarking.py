"""
Benchmarking runtime and memory usage.
Code uses MNIST_Classifier class (an RNN + FC layer), but otherwise is independent of any experiment.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
from time import time
import numpy as np
from mnist_classifier import MNIST_Classifier
from GPUtil import showUtilization as gpu_usage


def testing_eval():
    print("Benchmarking test time...")
    model.eval()
    all_durations = []
    _ = model(data)   # do one time to wake gpu.

    with torch.no_grad():
        for i in np.arange(args.nruns):
            # time forward pass
            start = time()
            _ = model(data)
            gpu_usage()
            duration = time() - start

            # save duration
            print(f"Run: {i} \t Duration: {duration}", )
            all_durations.append(duration)

    # print mean and std of durations.
    all_durations = np.array(all_durations)
    mean_time = np.mean(all_durations)
    std_time = np.std(all_durations)
    print(f"mean time: {mean_time} \t std time: {std_time}")


def training_eval():
    print("Benchmarking training time...")
    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=0.001)
    model.train()
    all_durations = []
    out = model(data)   # do one time to wake gpu.

    for i in np.arange(args.nruns):
        # start timer
        start = time()

        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()

        # end timer.
        gpu_usage()
        duration = time() - start

        # print and save duration
        print(f"Run: {i} \t Duration: {duration}", )
        all_durations.append(duration)

    # print mean and std of durations.
    all_durations = np.array(all_durations)
    mean_time = np.mean(all_durations)
    std_time = np.std(all_durations)
    print(f"mean time: {mean_time} \t std time: {std_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tt", action='store_true')
    parser.add_argument("--ncores", type=int, default=1)
    parser.add_argument("--ttrank", type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512, metavar='N')
    parser.add_argument("--in_size", type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=160)
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("-n", "--nruns", type=int, default=100)
    parser.add_argument("--gru", action='store_true')
    parser.add_argument("--naive_tt", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--cuda", action='store_true')
    args = parser.parse_args()
    print(args)

    # instantiate model
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Using the MNIST_Classifier class, but it's just a generic RNN with fully connected layer.
    model = MNIST_Classifier(args.in_size, args.emb_size, args.hidden_size, args.n_layers, device,
                             tt=args.tt, gru=args.gru, n_cores=args.ncores, tt_rank=args.ttrank,
                             naive_tt=args.naive_tt).to(device)

    # create random batch of data (using appropriate sizes)
    data = np.random.rand(args.batch_size, args.seq_len, args.in_size).astype('float32')
    target = np.random.randint(0, args.emb_size, args.batch_size).astype('int64')
    data = torch.from_numpy(data).to(device)
    target = torch.from_numpy(target).to(device)
    print("Benchmarking with input: {} and target {}".format(data.size(), target.shape))

    if args.train:
        training_eval()
    else:
        testing_eval()

    print(f"model has: {model.param_count()} parameters")

