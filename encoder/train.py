from .comet_logger import CometLogger

from pathlib import Path
import torch

from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset, \
    SpeakerVerificationTestSet, SpeakerVerificationTestDataLoader
from encoder.model import SpeakerEncoder
from utils.modelutils import count_model_params
from utils.ioutils import load_json, save_json
from .test import evaluate
from encoder import params_model as pm
from encoder import params_data as pd


def train(run_id: str, clean_data_root: Path, clean_data_root_val: Path, models_dir: Path,
        umap_every: int, val_every: int, resume_experiment: bool, prev_exp_key: str,
        no_comet: bool, gpu_no: int):

    # create comet logger.
    logger = CometLogger(no_comet, is_existing=resume_experiment, prev_exp_key=prev_exp_key)
    run_id = logger.get_experiment_key()

    # log or load training parameters.
    params_fpath, state_fpath, umap_dir = create_paths(models_dir, run_id)
    log_or_load_parameters(logger, resume_experiment, params_fpath)

    # setup dataset and model.
    train_loader, val_loader = create_dataloaders(clean_data_root, clean_data_root_val)
    device, loss_device = get_devices(gpu_no)
    model, optimizer, init_step, model_val_eer = \
        create_model_and_optimizer(device, loss_device, resume_experiment, state_fpath, run_id)

    if pm.use_tt:
        logger.add_tag("tt-cores{}-rank{}".format(pm.n_cores, pm.tt_rank))
    else:
        logger.add_tag("no-tt")

    # Training loop
    best_val_eer = model_val_eer
    print("Loading first batch for training...")
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
            # avg_val_loss, avg_val_eer = evaluate(val_loader, model, pm.test_speakers_per_batch,
            #                     pm.test_utterances_per_speaker, pm.test_n_epochs, device, loss_device)
            avg_val_loss, avg_val_eer = evaluate(val_loader, model, pm.test_speakers_per_batch,
                                        pm.test_utterances_per_speaker, device, loss_device)
            logger.log_metrics({"EER": avg_val_eer, "loss": avg_val_loss}, prefix="val", step=step)
            print("Step: {} - Validation Average loss: {}\t\tAverage EER: {}".
                  format(step, avg_val_loss, avg_val_eer))

            if avg_val_eer < best_val_eer:  # save current model
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

        break

def create_dataloaders(clean_data_root_train, clean_data_root_val):
    # set dataset length to achieve desired no of training steps.
    train_dataset_len = pm.n_steps * pm.speakers_per_batch

    # Create datasets and dataloaders
    train_loader = SpeakerVerificationDataLoader(
        SpeakerVerificationDataset(clean_data_root_train, train_dataset_len),
        pm.speakers_per_batch,
        pm.utterances_per_speaker,
        pd.partials_n_frames,
        num_workers=12,
        drop_last=True
    )

    val_loader = SpeakerVerificationTestDataLoader(
        SpeakerVerificationTestSet(clean_data_root_val),
        pm.test_speakers_per_batch,
        pm.test_utterances_per_speaker,
        pd.partials_n_frames,
        num_workers=12,
        drop_last=False
    )
    return train_loader, val_loader


def create_paths(models_dir, run_id):
    exp_dir = models_dir / run_id
    exp_dir.mkdir(exist_ok=True)

    umap_dir = exp_dir / "umap_pngs"
    umap_dir.mkdir(exist_ok=True)

    params_fpath = exp_dir / "params.txt"
    state_fpath = exp_dir / "model.pt"
    return params_fpath, state_fpath, umap_dir


def create_model_and_optimizer(device, loss_device, resume_experiment, state_fpath, run_id):
    # model
    model = SpeakerEncoder(pd.mel_n_channels, pm.model_hidden_size, pm.model_num_layers,
                           pm.model_embedding_size, device, loss_device,
                           use_tt=pm.use_tt, n_cores=pm.n_cores, tt_rank=pm.tt_rank)
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

def log_or_load_parameters(logger, resume_experiment, params_fpath):
    if resume_experiment:
        if not params_fpath.exists():
            raise FileNotFoundError("Cannot resume experiment. No parameters file '{}' found"
                                    .format(params_fpath))
        params = load_json(params_fpath)
        for param_name in (p for p in dir(pm) if not p.startswith("__")):
            setattr(pm, param_name, params[param_name])
        for param_name in (p for p in dir(pd) if not p.startswith("__")):
            setattr(pd, param_name, params[param_name])
        print("Loaded existing parameters from: {}".format(params_fpath))
    else:
        logger.log_params(params_fpath)
        print("Saved parameters to: {}".format(params_fpath))
