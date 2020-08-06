"""
Benchmarking runtime of speaker verfication models.
"""

import torch
import argparse
from time import time
import numpy as np
from encoder.speaker_encoder import SpeakerEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--compression", type=str, default=None)
    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("-n", "--n_runs", type=int, default=100)
    args = parser.parse_args()
    print(args)

    # instantiate model
    device = torch.device('cuda')
    loss_device = torch.device('cpu')
    model = SpeakerEncoder(mel_n_channels=40, model_hidden_size=768, model_num_layers=1,
                    model_embedding_size=256, device=device, loss_device=loss_device,
                    compression=args.compression, n_cores=args.n_cores, rank=args.rank)

    # create random batch of data (using appropriate sizes)
    random_batch = np.random.rand(16 * 32, 160, 40).astype('float32')
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
            duration = time() - start
            print(f"Run: {i} \t Duration: {duration}", )
            all_durations.append(duration)
    all_durations = np.array(all_durations)
    mean_time = np.mean(all_durations)
    std_time = np.std(all_durations)
    print(f"mean time: {mean_time} \t std time: {std_time}")
