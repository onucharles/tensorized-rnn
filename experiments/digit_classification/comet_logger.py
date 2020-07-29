import pickle
import warnings

from comet_ml import Experiment, ExistingExperiment
import matplotlib.pyplot as plt
import numpy as np

from config_j import COMET_API_KEY, COMET_WORKSPACE, PROJECT_NAME
# from config import COMET_API_KEY, COMET_WORKSPACE, PROJECT_NAME

class CometLogger():
    def __init__(self, disabled, is_existing=False, prev_exp_key=None):
        """
        Handles logging of experiment to comet and also persistence to local file system.
        Supports resumption of stopped experiments.
        """

        if not is_existing:
            self.experiment = Experiment(api_key=COMET_API_KEY,
                                         workspace=COMET_WORKSPACE,
                                         project_name=PROJECT_NAME,
                                         disabled=disabled)
        else:
            if prev_exp_key is None:
                raise ValueError("Requested existing experiment, but no key provided")
            print("Continuing existing experiment with key: ", prev_exp_key)
            self.experiment = ExistingExperiment(api_key=COMET_API_KEY,
                                                 workspace=COMET_WORKSPACE,
                                                 project_name=PROJECT_NAME,
                                                 disabled=disabled,
                                                 previous_experiment=prev_exp_key)
        self.disabled = disabled
        self.name = None

    def get_experiment_key(self):
        return self.experiment.get_key()[:9]

    def add_tag(self, tag):
        self.experiment.add_tag(tag)

    def log_metric(self, name, value, step=None):
        self.experiment.log_metric(name, value, step=step)

    def log_metrics(self, metrics_dict, prefix, step=None):
        self.experiment.log_metrics(metrics_dict, prefix=prefix, step=step)

    def log_params(self, params_dict):
        self.experiment.log_parameters(params_dict)

    def set_name(self, name_str):
        self.experiment.set_name(name_str)
        self.name = name_str

    def save_act_grads(self, log_dict):
        """Save a dictionary of activation/gradients records to disk"""
        assert isinstance(log_dict, dict)
        if self.name is None:
            warnings.warn("Experiment name not set, not saving")
            return

        # Save the log dictionary
        file_name = f"./.{self.name}.record"
        with open(file_name, 'wb') as f:
            pickle.dump(log_dict, f)

    # TODO: need to rewrite before can be used for MNIST.
    def draw_projections(self, embeds, utterances_per_speaker, step, out_fpath=None,
                         max_speakers=16):
        import umap
        if self.disabled:
            return
        max_speakers = min(max_speakers, len(colormap))
        embeds = embeds[:max_speakers * utterances_per_speaker]

        n_speakers = len(embeds) // utterances_per_speaker
        ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
        colors = [colormap[i] for i in ground_truth]

        reducer = umap.UMAP()
        projected = reducer.fit_transform(embeds)
        plt.scatter(projected[:, 0], projected[:, 1], c=colors)
        plt.gca().set_aspect("equal", "datalim")
        plt.title("UMAP projection (step %d)" % step)
        if out_fpath is not None:
            plt.savefig(out_fpath)
        plt.clf()
        self.experiment.log_image(out_fpath, step=step)

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
], dtype=np.float) / 255
