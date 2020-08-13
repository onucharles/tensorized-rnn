"""
Benchmarking runtime of speaker verfication models.
"""

import torch
import argparse
from time import time
import numpy as np
from encoder.speaker_encoder import SpeakerEncoder
from GPUtil import showUtilization as gpu_usage
from utils.modelutils import count_model_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--compression", type=str, default=None)
    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("-n", "--n_runs", type=int, default=100)
    parser.add_argument("--in_size", type=int, default=40)
    parser.add_argument("--h_size", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=160)
    args = parser.parse_args()
    print(args)

    # instantiate model
    device = torch.device('cuda')
    loss_device = torch.device('cpu')
    model = SpeakerEncoder(mel_n_channels=args.in_size,
            model_hidden_size=args.h_size, model_num_layers=args.n_layers,
                model_embedding_size=args.emb_size, device=device, loss_device=loss_device,
                    compression=args.compression, n_cores=args.n_cores, rank=args.rank)

    # create random batch of data (using appropriate sizes)
    random_batch = np.random.rand(args.batch_size, args.seq_len, args.in_size).astype('float32')
    random_batch = torch.from_numpy(random_batch).to(device)
    print("Benchmarking with input of shape (n_speakers * n_utterances, seq_length, input_channels): ",
          random_batch.shape)

    # run and time forward pass x times.
    model.eval()
    all_durations = []
    out = model(random_batch)   # do one time to activate gpu.
    with torch.no_grad():
        for i in np.arange(args.n_runs):
            start = time()
            out = model(random_batch)
            gpu_usage()
            duration = time() - start
            print(f"Run: {i} \t Duration: {duration}", )
            all_durations.append(duration)
    all_durations = np.array(all_durations)
    mean_time = np.mean(all_durations)
    std_time = np.std(all_durations)
    print(f"mean time: {mean_time} \t std time: {std_time}")
    print("Model has {} parameters".format(count_model_params(model)))
