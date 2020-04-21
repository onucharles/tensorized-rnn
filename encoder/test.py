from encoder.visualizations import Visualizations
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset, SpeakerVerificationTestSet
from encoder.params_model import test_speakers_per_batch, test_utterances_per_speaker
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler
from pathlib import Path
import torch
import numpy as np

_model = None # type: SpeakerEncoder
_device = None # type: torch.device

def test(test_data_dir: Path, model_path: Path, n_workers: int):

    # create dataset and data loader
    dataset = SpeakerVerificationTestSet(test_data_dir)
    loader = SpeakerVerificationDataLoader(
        dataset=dataset,
        speakers_per_batch=test_speakers_per_batch,
        utterances_per_speaker=test_utterances_per_speaker,
        num_workers=n_workers,
        drop_last=True
    )

    # get cuda device if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _device = torch.device("cuda")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")

    # initialise and load model
    _model = SpeakerEncoder(_device, loss_device)
    if model_path.exists():
        print("Loading model at: {}".format(model_path))
        checkpoint = torch.load(model_path)
        _model.load_state_dict(checkpoint["model_state"])
        _model.eval()
    else:
        print("No model found to load. Exiting...")
        return

    print("Running test with speakers_per_batch: {}, utterances_per_speaker: {}"
            .format(test_speakers_per_batch, test_utterances_per_speaker))

    avg_loss, avg_eer = 0, 0
    n_epochs = 10
    for epoch in range(n_epochs):
        for step, speaker_batch in enumerate(loader):
            #print("---------Step {}----------".format(step))

            inputs = torch.from_numpy(speaker_batch.data).to(_device)    # shape: (n_speakers * n_utter, n_frames, n_mels)
            embeddings = _model(inputs)  # shape: (n_speakers * n_utter, d_vector_size)
            embeddings = embeddings.view((test_speakers_per_batch, test_utterances_per_speaker, -1)).to(loss_device)    # shape: (n_speakers, n_utter, d_vector_size)

            # split each speakers' utterances into enrollment and verification sets.
            verification_embeds, enrollment_embeds = torch.chunk(embeddings, 2, dim=1)

            loss, eer = _model.loss(verification_embeds, enrollment_embeds)
            #print("loss: {}\tEER: {}".format(loss, eer))

            avg_loss += loss.item()
            avg_eer += eer

    avg_loss /= (step + 1) * n_epochs
    avg_eer /= (step + 1) * n_epochs
    print("Average loss: {}\t\tAverage EER: {}".format(avg_loss, avg_eer))
    return avg_loss, avg_eer
