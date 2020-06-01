from .comet_logger import CometLogger

from pathlib import Path
import torch
import numpy as np
import random
import warnings
import json
from torchsummary import summary

from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset, \
    SpeakerVerificationTestSet, SpeakerVerificationTestDataLoader
from encoder.models.speaker_encoder import SpeakerEncoder
from utils.modelutils import count_model_params
from utils.ioutils import load_json
from encoder import params_model as pm
from encoder import params_data as pd


def train(clean_data_root: Path, clean_data_root_val: Path, models_dir: Path,
        umap_every: int, val_every: int, resume_experiment: bool, prev_exp_key: str,
        no_comet: bool, gpu_no: int, seed: int):
    """
    Main entry point for training.
    """

    # create comet logger.
    logger = CometLogger(no_comet, is_existing=resume_experiment, prev_exp_key=prev_exp_key)
    run_id = logger.get_experiment_key()

    # log or load training parameters.
    params_fpath, state_fpath, umap_dir = create_paths(models_dir, run_id)
    log_or_load_parameters(logger, resume_experiment, params_fpath, no_comet)

    # setup dataset and model.
    set_seed(seed)
    train_loader = create_train_loader(clean_data_root)
    val_loader = create_test_loader(clean_data_root_val, pm.val_speakers_per_batch,
                                    pm.val_utterances_per_speaker, pd.partials_n_frames)
    device, loss_device = get_devices(gpu_no)
    model, optimizer, init_step, model_val_eer = \
        create_model_and_optimizer(device, loss_device, resume_experiment, state_fpath, run_id)

    if pm.compression == 'tt':
        logger.add_tag("tt-cores{}-rank{}".format(pm.n_cores, pm.rank))
    elif pm.compression == 'lr':
        logger.add_tag("low-rank{}".format(pm.rank))
    elif pm.compression is None:
        logger.add_tag("no-comp")
    else:
        raise ValueError('Unknown compression value: "{}"'.format(pm.compression))

    # Training loop
    best_val_eer = model_val_eer
    print("Starting training loop...")
    for step, speaker_batch in enumerate(train_loader, init_step):
        model.train()

        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        embeds = model(inputs)
        embeds_loss = embeds.view((pm.speakers_per_batch, pm.utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)

        # Backward pass
        model.zero_grad()
        loss.backward()
        model.do_gradient_ops()
        optimizer.step()

        logger.log_metrics({"EER": eer, "loss": loss.item()}, prefix="train", step=step)
        print("Step: {}\tTrain Loss: {}\tTrain EER: {}".format(step, loss.item(), eer))

        if val_every != 0 and step % val_every == 0:
            avg_val_loss, avg_val_eer = evaluate(val_loader, model, pm.val_speakers_per_batch,
                                        pm.val_utterances_per_speaker, device, loss_device)
            logger.log_metrics({"EER": avg_val_eer, "loss": avg_val_loss}, prefix="val", step=step)
            print("Step: {} - Validation Average loss: {}\t\tAverage EER: {}".
                  format(step, avg_val_loss, avg_val_eer))

            if no_comet: continue
            if best_val_eer - avg_val_eer > 1e-4:  # save current model if improvement is significant
                print("Saving the model (step %d)" % step)
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_eer": avg_val_eer,
                }, state_fpath)
                best_val_eer = avg_val_eer
                logger.log_metric("best_eer", best_val_eer)

        # Draw projections and save them to the backup folder
        if umap_every != 0 and step % umap_every == 0:
            print("Drawing and saving projections (step %d)" % step)
            projection_fpath = umap_dir / ("%s_umap_%06d.png" % (run_id, step))
            embeds = embeds.detach().cpu().numpy()
            logger.draw_projections(embeds, pm.utterances_per_speaker, step, projection_fpath)


def test(test_data_dir: Path, exp_root_dir: Path, prev_exp_key: str,
         no_comet: bool, gpu_no: int):
    """
    Main entry point for testing.
    """
    logger = CometLogger(no_comet, is_existing=True, prev_exp_key=prev_exp_key)
    run_id = logger.get_experiment_key()

    # create loader.
    speakers_per_batch = pm.test_speakers_per_batch
    utterances_per_speaker = pm.test_utterances_per_speaker
    print("Testing with {} speakers per batch and {} utterances per speaker."
          .format(speakers_per_batch, utterances_per_speaker))
    test_loader = create_test_loader(test_data_dir, speakers_per_batch,
                                     utterances_per_speaker, pd.partials_n_frames)

    # initialise and load model
    params_fpath, state_fpath, _ = create_paths(exp_root_dir, run_id)
    log_or_load_parameters(logger, resume_experiment=True, params_fpath=params_fpath)
    device, loss_device = get_devices(gpu_no)
    model = SpeakerEncoder(pd.mel_n_channels, pm.model_hidden_size, pm.model_num_layers,
                           pm.model_embedding_size, device, loss_device, use_low_rank=pm.use_low_rank,
                           use_tt=pm.use_tt, n_cores=pm.n_cores, rank=pm.rank)
    if state_fpath.exists():
        print("Found existing model \"%s\", loading it." % state_fpath)
        checkpoint = torch.load(state_fpath)
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise FileNotFoundError("No model \"%s\" found." % run_id)

    # evaluate the model on test data.
    avg_loss, avg_eer = evaluate(test_loader, model, speakers_per_batch,
                                 utterances_per_speaker, device, loss_device)
    print("Test - loss: {}\t\t EER: {}".format(avg_loss, avg_eer))
    logger.log_metrics({"EER": avg_eer, "loss": avg_loss}, prefix="test")

# ------------------- Training and Testing Helpers ---------------- #


def evaluate(loader, model, speakers_per_batch, utterances_per_speaker, device, loss_device):
    model.eval()
    with torch.no_grad():
        enrollment_embeds = []
        verification_embeds = []
        for step, speaker_batch in enumerate(loader):
            for speaker, (frame_slices_in_utterances, slice_counts) in speaker_batch.data.items():
                # run model
                inputs = torch.from_numpy(np.concatenate(frame_slices_in_utterances)).to(
                    device)  # shape: (n_slices, n_frames, n_mels)
                embeddings = model(inputs)  # shape: (n_slices, d_vector_size)
                utterances_embeddings = torch.split(embeddings, slice_counts,
                                                    0)  # tuple of len = no of utterances in speaker. where each is (m_slices, d_vector_size)

                # get average utterance embedding.
                mean_utterances_embedding = [torch.mean(utter_emb, dim=0, keepdim=True)
                                             for utter_emb in
                                             utterances_embeddings]  # List of <n_utters> items of shape (1, d_vector_size)

                # split utterance embeddings into enrollment and verification sets.
                assert len(utterances_embeddings) == utterances_per_speaker
                n_enrol = len(utterances_embeddings) // 2
                enrollment_embeds += mean_utterances_embedding[
                                     :n_enrol]  # List of <n_enrol> items of shape (1, d_vector_size)
                verification_embeds += mean_utterances_embedding[n_enrol:]
        # concatenate embeddings for all speakers and reshape.
        n_speakers = len(enrollment_embeds) // n_enrol
        enrollment_embeds = torch.cat(enrollment_embeds).view((n_speakers, n_enrol, -1))
        verification_embeds = torch.cat(verification_embeds).view((n_speakers, n_enrol, -1))
        loss, eer = model.loss(verification_embeds.to(loss_device), enrollment_embeds.to(loss_device))

    return loss.item(), eer

def create_train_loader(clean_data_root):
    # set dataset length to achieve desired no of training steps.
    train_dataset_len = pm.n_steps * pm.speakers_per_batch

    # Create datasets and dataloaders
    train_loader = SpeakerVerificationDataLoader(
        SpeakerVerificationDataset(clean_data_root, train_dataset_len),
        pm.speakers_per_batch,
        pm.utterances_per_speaker,
        pd.partials_n_frames,
        num_workers=12,
        drop_last=True
    )
    return train_loader


def create_test_loader(clean_data_root, speakers_per_batch, utterances_per_speaker,
                       partials_n_frames):
    test_loader = SpeakerVerificationTestDataLoader(
        SpeakerVerificationTestSet(clean_data_root),
        speakers_per_batch,
        utterances_per_speaker,
        partials_n_frames,
        num_workers=12,
        drop_last=False
    )
    return test_loader


def create_paths(models_dir, run_id, no_logging=False):
    exp_dir = models_dir / run_id
    umap_dir = exp_dir / "umap_pngs"

    if not no_logging:
        exp_dir.mkdir(exist_ok=True)
        umap_dir.mkdir(exist_ok=True)

    params_fpath = exp_dir / "params.txt"
    state_fpath = exp_dir / "model.pt"
    return params_fpath, state_fpath, umap_dir


def create_model_and_optimizer(device, loss_device, resume_experiment, state_fpath, run_id):
    # model
    model = SpeakerEncoder(pd.mel_n_channels, pm.model_hidden_size, pm.model_num_layers,
                           pm.model_embedding_size, device, loss_device,
                           compression=pm.compression, n_cores=pm.n_cores,
                           rank=pm.rank)
    # summary(model, (pd.partials_n_frames, pd.mel_n_channels))
    n_trainable, n_nontrainable = count_model_params(model)
    print("Model instantiated. Trainable params: {}, Non-trainable params: {}. Total: {}"
          .format(n_trainable, n_nontrainable, n_trainable + n_nontrainable))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=pm.learning_rate_init)
    init_step = 1
    model_val_eer = 1.0

    # Load any existing model
    if resume_experiment:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % state_fpath)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model_val_eer = checkpoint["val_eer"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = pm.learning_rate_init
        else:
            raise FileNotFoundError("Cannot resume experiment. No model \"%s\" found." % run_id)
    else:
        print("Starting the training from scratch.")
    return model, optimizer, init_step, model_val_eer


def get_devices(gpu_no):
    # Setup the device on which to run the forward pass and the loss. These can be different,
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    gpu_name = 'cuda:{}'.format(gpu_no)
    device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")
    return device, loss_device

def log_or_load_parameters(logger, resume_experiment, params_fpath, no_logging=False):
    if resume_experiment:   # load parameters
        if not params_fpath.exists():
            raise FileNotFoundError("Cannot resume experiment. No parameters file '{}' found"
                                    .format(params_fpath))
        params = load_json(params_fpath)
        for param_name in (p for p in dir(pm) if not p.startswith("__")):
            if param_name in params:
                setattr(pm, param_name, params[param_name])
            else:
                warnings.warn("Unable to load parameter: '{}'. Not found.".format(param_name))
        for param_name in (p for p in dir(pd) if not p.startswith("__")):
            if param_name in params:
                setattr(pd, param_name, params[param_name])
            else:
                warnings.warn("Unable to load parameter: '{}'. Not found.".format(param_name))
        print("Loaded existing parameters from: {}".format(params_fpath))
    else:   # log parameters
        # logger.log_params(params_fpath)
        if no_logging:
            return

        parameters = {}
        for param_name in (p for p in dir(pm) if not p.startswith("__")):
            value = getattr(pm, param_name)
            parameters[param_name] = value

        for param_name in (p for p in dir(pd) if not p.startswith("__")):
            value = getattr(pd, param_name)
            parameters[param_name] = value

        # log to comet and save to file.
        logger.log_params(parameters)
        with open(params_fpath, 'w') as fp:
            json.dump(parameters, fp, sort_keys=True, indent=4)
        print("Saved parameters to: {}".format(params_fpath))


def set_seed(seed=None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
