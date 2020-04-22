from .comet_logger import CometLogger
from encoder.data_objects import \
    SpeakerVerificationDataLoader, SpeakerVerificationDataset, SpeakerVerificationTestSet
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler
from utils.modelutils import count_model_params
from pathlib import Path
import torch
from .test import evaluate


# def train(run_id: str, clean_data_root: Path, clean_data_root_val: Path, models_dir: Path,
#           umap_every: int, save_every: int, backup_every: int, val_every: int,
#           force_restart: bool, no_comet: bool):
def train(run_id: str, clean_data_root: Path, clean_data_root_val: Path, models_dir: Path,
          umap_every: int, val_every: int, force_restart: bool, no_comet: bool):
    # create comet logger.
    logger = CometLogger(no_comet)
    run_id = logger.get_key()

    # setup data and model.
    train_loader, val_loader = create_dataloaders(clean_data_root, clean_data_root_val)
    device, loss_device = get_devices()
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")
    model, optimizer, init_step = \
        setup_model_and_optimizer(device, loss_device, force_restart, state_fpath, run_id)

    # Training loop
    best_val_eer = 1
    profiler = Profiler(summarize_every=10, disabled=True)
    for step, speaker_batch in enumerate(train_loader, init_step):
        model.train()
        profiler.tick("Blocking, waiting for batch (threaded)")

        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        profiler.tick("Data to %s" % device)
        embeds = model(inputs)
        sync(device)
        profiler.tick("Forward pass")
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        profiler.tick("Loss")

        # Backward pass
        model.zero_grad()
        loss.backward()
        profiler.tick("Backward pass")
        model.do_gradient_ops()
        optimizer.step()
        profiler.tick("Parameter update")

        # Update visualizations
        logger.log_metrics({"EER": eer, "loss": loss.item()}, prefix="train", step=step)
        # print("Step: {}\tTrain Loss: {}\tTrain EER: {}".format(step, loss.item(), eer))

        if val_every != 0 and step % val_every == 0:
            avg_val_loss, avg_val_eer = evaluate(val_loader, model, device, loss_device)
            logger.log_metrics({"EER": avg_val_eer, "loss": avg_val_loss}, prefix="val", step=step)
            print("Step: {} - Validation Average loss: {}\t\tAverage EER: {}".format(step, avg_val_loss, avg_val_eer))

            if avg_val_eer < best_val_eer:  # save current model
                print("Saving the model (step %d)" % step)
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, state_fpath)

        # Draw projections and save them to the backup folder
        if umap_every != 0 and step % umap_every == 0:
            print("Drawing and saving projections (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
            embeds = embeds.detach().cpu().numpy()
            logger.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)

        # # Overwrite the latest version of the model
        # if save_every != 0 and step % save_every == 0:
        #     print("Saving the model (step %d)" % step)
        #     torch.save({
        #         "step": step + 1,
        #         "model_state": model.state_dict(),
        #         "optimizer_state": optimizer.state_dict(),
        #     }, state_fpath)
        #
        # # Make a backup
        # if backup_every != 0 and step % backup_every == 0:
        #     print("Making a backup (step %d)" % step)
        #     backup_dir.mkdir(exist_ok=True)
        #     backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
        #     torch.save({
        #         "step": step + 1,
        #         "model_state": model.state_dict(),
        #         "optimizer_state": optimizer.state_dict(),
        #     }, backup_fpath)

        profiler.tick("Extras (visualizations, saving)")


def sync(device: torch.device):
    # FIXME
    return
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def create_dataloaders(clean_data_root_train, clean_data_root_val):
    # Create a dataset and a dataloader
    train_loader = SpeakerVerificationDataLoader(
        SpeakerVerificationDataset(clean_data_root_train, n_epochs),
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=10,
    )

    val_loader = SpeakerVerificationDataLoader(
        dataset=SpeakerVerificationTestSet(clean_data_root_val),
        speakers_per_batch=test_speakers_per_batch,
        utterances_per_speaker=test_utterances_per_speaker,
        num_workers=10,
        drop_last=True
    )
    return train_loader, val_loader


def setup_model_and_optimizer(device, loss_device, force_restart, state_fpath, run_id):
    # Create the model and the optimizer
    model = SpeakerEncoder(device, loss_device, use_tt=use_tt, n_cores=n_cores, tt_rank=tt_rank)
    n_trainable, n_nontrainable = count_model_params(model)
    print("Model created. Trainable params: {}, Non-trainable params: {}. Total: {}"
          .format(n_trainable, n_nontrainable, n_trainable + n_nontrainable))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")
    return model, optimizer, init_step


def get_devices():
    # Setup the device on which to run the forward pass and the loss. These can be different,
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")
    return device, loss_device
