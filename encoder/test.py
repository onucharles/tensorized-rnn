from pathlib import Path
import torch
import numpy as np

from encoder.data_objects import SpeakerVerificationTestSet, SpeakerVerificationTestDataLoader
from encoder.model import SpeakerEncoder
from encoder import params_model as pm
from encoder import params_data as pd

def evaluate(loader, model, speakers_per_batch, utterances_per_speaker, device, loss_device):

    model.eval()
    with torch.no_grad():
        enrollment_embeds = []
        verification_embeds = []
        for step, speaker_batch in enumerate(loader):
            for speaker, (frame_slices_in_utterances, slice_counts) in speaker_batch.data.items():
                print("slice_counts: ", slice_counts)
                print("frame_slices_in_utterances count {}. Shape of one item: {}"
                      .format(len(frame_slices_in_utterances), frame_slices_in_utterances[0].shape))

                inputs = torch.from_numpy(np.concatenate(frame_slices_in_utterances)).to(device)  # shape: (n_slices, n_frames, n_mels)
                embeddings = model(inputs)      # shape: (n_slices, d_vector_size)
                utterances_embeddings = torch.split(embeddings, slice_counts, 0)     # tuple of len = no of utterances in speaker. where each is (m_slices, d_vector_size)
                print("len of utterances_embedding: {}. Shape of one item: {}".format(len(utterances_embeddings), utterances_embeddings[0].shape))

                # get average utterance embedding.
                mean_utterances_embedding = [torch.mean(utter_emb, dim=0, keepdim=True)
                                            for utter_emb in utterances_embeddings]    # List of <n_utters> items of shape (1, d_vector_size)
                print("len of mean_utterances_embedding: {}. Shape of one item: {}".format(len(mean_utterances_embedding),
                                                                                      mean_utterances_embedding[0].shape))

                # split utterance embeddings into enrollment and verification sets.
                assert len(utterances_embeddings) == utterances_per_speaker
                n_enrol = len(utterances_embeddings) // 2
                enrollment_embeds += mean_utterances_embedding[:n_enrol]   # List of <n_enrol> items of shape (1, d_vector_size)
                verification_embeds += mean_utterances_embedding[n_enrol:]

        print("len of enrollment_embeds: {}. Shape of one item: {}".format(len(enrollment_embeds),
                                                                                   enrollment_embeds[0].shape))

        # concatenate embeddings for all speakers and reshape.
        n_speakers = len(enrollment_embeds) // n_enrol
        enrollment_embeds = torch.cat(enrollment_embeds).view((n_speakers, n_enrol, -1))
        verification_embeds = torch.cat(verification_embeds).view((n_speakers, n_enrol, -1))
        print("shape of enrollment embeds: {}".format(enrollment_embeds.shape))
        loss, eer = model.loss(verification_embeds.to(loss_device), enrollment_embeds.to(loss_device))
        print("loss: {}\tEER: {}".format(loss, eer))

    return loss.item(), eer

# def evaluate(loader, model, speakers_per_batch, utterances_per_speaker, n_epochs, device, loss_device):
#     avg_loss, avg_eer = 0, 0
#
#     # Go over entire test set for a few epochs. Important since <test_utterances_per_speaker>,
#     # will likely be smaller than the average no of utterances per speaker.
#     model.eval()
#     with torch.no_grad():
#         for epoch in range(n_epochs):
#             for step, speaker_batch in enumerate(loader):
#                 # print("---------Step {}----------".format(step))
#                 inputs = torch.from_numpy(speaker_batch.data).to(device)  # shape: (n_speakers * n_utter, n_frames, n_mels)
#                 embeddings = model(inputs)  # shape: (n_speakers * n_utter, d_vector_size)
#                 embeddings = embeddings.view((speakers_per_batch, utterances_per_speaker, -1))  # shape: (n_speakers, n_utter, d_vector_size)
#
#                 # split each speakers' utterances into enrollment and verification sets.
#                 verification_embeds, enrollment_embeds = torch.chunk(embeddings, 2, dim=1)
#
#                 loss, eer = model.loss(verification_embeds.to(loss_device), enrollment_embeds.to(loss_device))
#                 # print("loss: {}\tEER: {}".format(loss, eer))
#
#                 avg_loss += loss.item()
#                 avg_eer += eer
#
#     avg_loss /= (step + 1) * n_epochs
#     avg_eer /= (step + 1) * n_epochs
#     return avg_loss, avg_eer

def test(test_data_dir: Path, model_path: Path, n_workers: int):

    # create dataset and data loader
    print("Running test with speakers_per_batch: {}, utterances_per_speaker: {}"
            .format(pm.test_speakers_per_batch, pm.test_utterances_per_speaker))
    dataset = SpeakerVerificationTestSet(test_data_dir)
    loader = SpeakerVerificationTestDataLoader(
        dataset,
        pm.test_speakers_per_batch,
        pm.test_utterances_per_speaker,
        pd.partials_n_frames,
        num_workers=n_workers,
        drop_last=False
    )

    # get cuda device if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")

    # initialise and load model
    model = SpeakerEncoder(pd.mel_n_channels, pm.model_hidden_size, pm.model_num_layers,
                           pm.model_embedding_size, device, loss_device,
                           use_tt=pm.use_tt, n_cores=pm.n_cores, tt_rank=pm.tt_rank)
    if model_path.exists():
        print("Loading model at: {}".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state"])
    else:
        print("No model found to load. Exiting...")
        return

    # evaluate the model.
    avg_loss, avg_eer = evaluate(loader, model, pm.test_speakers_per_batch,
                                 pm.test_utterances_per_speaker, device, loss_device)
    print("Average loss: {}\t\tAverage EER: {}".format(avg_loss, avg_eer))


