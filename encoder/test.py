from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationTestSet
from encoder.params_model import test_speakers_per_batch, test_utterances_per_speaker, test_n_epochs
from encoder.model import SpeakerEncoder
from pathlib import Path
import torch

def evaluate(loader, model, device, loss_device):
    avg_loss, avg_eer = 0, 0
    n_epochs = test_n_epochs

    # Go over entire test set for a few epochs. Important since <test_utterances_per_speaker>,
    # will likely be smaller than the average no of utterances per speaker.
    model.eval()
    with torch.no_grad():
        for epoch in range(n_epochs):
            for step, speaker_batch in enumerate(loader):
                # print("---------Step {}----------".format(step))
                inputs = torch.from_numpy(speaker_batch.data).to(device)  # shape: (n_speakers * n_utter, n_frames, n_mels)
                embeddings = model(inputs)  # shape: (n_speakers * n_utter, d_vector_size)
                embeddings = embeddings.view((test_speakers_per_batch, test_utterances_per_speaker, -1)).to(
                    loss_device)  # shape: (n_speakers, n_utter, d_vector_size)

                # split each speakers' utterances into enrollment and verification sets.
                verification_embeds, enrollment_embeds = torch.chunk(embeddings, 2, dim=1)

                loss, eer = model.loss(verification_embeds, enrollment_embeds)
                # print("loss: {}\tEER: {}".format(loss, eer))

                avg_loss += loss.item()
                avg_eer += eer

    avg_loss /= (step + 1) * n_epochs
    avg_eer /= (step + 1) * n_epochs
    return avg_loss, avg_eer

def test(test_data_dir: Path, model_path: Path, n_workers: int):

    # create dataset and data loader
    print("Running test with speakers_per_batch: {}, utterances_per_speaker: {}"
            .format(test_speakers_per_batch, test_utterances_per_speaker))
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
    device = torch.device("cuda")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")

    # initialise and load model
    model = SpeakerEncoder(device, loss_device)
    if model_path.exists():
        print("Loading model at: {}".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state"])
    else:
        print("No model found to load. Exiting...")
        return

    # evaluate the model.
    avg_loss, avg_eer = evaluate(loader, model, device, loss_device)
    print("Average loss: {}\t\tAverage EER: {}".format(avg_loss, avg_eer))


