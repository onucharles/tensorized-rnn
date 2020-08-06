"""
Visualise utterance embeddings in 2D using a given model and a subset of speaker utterances
"""

import umap
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.speaker_encoder import SpeakerEncoder
from encoder.data_objects.speaker_batch import SpeakerBatch

colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [183, 0, 111],
    [183, 12, 0],
    [0, 183, 1],
    [183, 183, 183],
    [183, 118, 225],
    [34, 183, 183],
    [183, 0, 79],
], dtype=np.float) / 255

def load_model(model_dir, device):
    # load model config and model.
    config = {}
    with open(model_dir / "params.txt", 'r') as fp:
        config = json.load(fp)

    model = SpeakerEncoder(config['mel_n_channels'], config['model_hidden_size'], config['model_num_layers'],
                           config['model_embedding_size'], device, torch.device('cpu'),
                           compression=config['compression'], n_cores=config['n_cores'],
                           rank=config['rank'])

    # load model state
    print('loading model state...')
    checkpoint = torch.load(model_dir / "model.pt", map_location='cuda:0')
    model.load_state_dict(checkpoint["model_state"])
    return model

def set_seed(seed=None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_batch_of_embeddings(data_dir, model_dir, n_speakers, n_utterances):
    # create data set
    speaker_dataset = SpeakerVerificationDataset(data_dir, dataset_len=26)

    # create model
    device = torch.device('cuda')
    model = load_model(model_dir, device)
    model.eval()

    # get embeddings
    print("running model on batch...")
    with torch.no_grad():
        speakers = [speaker_dataset[i] for i in np.random.randint(0, len(speaker_dataset), n_speakers)]# for i in np.arange(n_speakers)]
        my_batch = SpeakerBatch(speakers, utterances_per_speaker=n_utterances, n_frames=160)
        my_batch_data = torch.from_numpy(my_batch.data).to(device)
        print("Batch shape: ", my_batch_data.shape)
        embeds = model(my_batch_data)
        print("Embeds shape: ", embeds.shape)

        embeds = embeds.detach().cpu().numpy()
    return embeds

def draw_projections(embeds, n_speakers, utterances_per_speaker, out_fpath=None):
    max_speakers = min(n_speakers, len(colormap))
    embeds = embeds[:max_speakers * utterances_per_speaker]
    print('number of speakers being visualised: ', max_speakers)

    # n_speakers = len(embeds) // utterances_per_speaker
    ground_truth = np.repeat(np.arange(max_speakers), utterances_per_speaker)
    colors = np.array([colormap[i] for i in ground_truth])

    reducer = umap.UMAP()
    projected = reducer.fit_transform(embeds)
    plt.figure(dpi=200)

    for spk_id in np.arange(max_speakers):
        spk_idx = ground_truth == spk_id
        plt.scatter(projected[spk_idx, 0], projected[spk_idx, 1], c=colors[spk_idx], alpha=0.5, label=f"spk {spk_id}",
                    s=40)
    plt.gca().set_aspect("equal", "datalim")
    if out_fpath is not None:
        plt.savefig(out_fpath)
    plt.xlabel("dim 1", fontsize=13)
    plt.ylabel("dim 2", fontsize=13)
    plt.legend(fontsize=11)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    test_data_dir = Path(
        "..\..\..\_experiments\speech-model-compression\speaker-verification\_librispeech_train-clean-100_tisv")
    model_dir = Path(r"trained_models\tt-lstm")
    n_speakers = 12
    n_utterances = 20

    # for seed in [820]: #np.random.randint(0, 2000, 100):
    seed = 820
    set_seed(seed)
    embeddings = get_batch_of_embeddings(test_data_dir, model_dir, n_speakers, n_utterances)
    draw_projections(embeddings, n_speakers, n_utterances, out_fpath=None)
                     # out_fpath=Path(f"trained_models/tt-lstm/umap/{seed}.png"))


