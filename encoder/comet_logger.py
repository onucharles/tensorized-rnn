from comet_ml import Experiment
import matplotlib.pyplot as plt
import numpy as np
import umap
import json

from encoder.data_objects.speaker_verification_dataset import SpeakerVerificationDataset
from .config import COMET_API_KEY, COMET_WORKSPACE, PROJECT_NAME

class CometLogger(Experiment):
    def __init__(self, disabled, ):
        """
        Handles logging of experiment to comet and also persistence to local file system.
        """
        super().__init__(api_key=COMET_API_KEY,
                         workspace=COMET_WORKSPACE,
                         project_name=PROJECT_NAME,
                         disabled=disabled)
        self.disabled = disabled
        self.parameters = {}
        # log dataset.
        # log device name?? in case of multi gpu setup.

    def log_params(self, params_path):
        """
        Log data and model parameters to comet.
        :return:
        """
        if self.disabled:
            return
        from encoder import params_data
        from encoder import params_model
        for param_name in (p for p in dir(params_model) if not p.startswith("__")):
            value = getattr(params_model, param_name)
            self.parameters[param_name] = value

        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            self.parameters[param_name] = value
        self.log_parameters(self.parameters)

        # save to file.
        with open(params_path, 'w') as fp:
            json.dump(self.parameters, fp, sort_keys=True, indent=4)

    def log_dataset(self, dataset: SpeakerVerificationDataset):
        if self.disabled:
            return
        dataset_string = ""
        dataset_string += "<b>Speakers</b>: %s\n" % len(dataset.speakers)
        dataset_string += "\n" + dataset.get_logs()
        dataset_string = dataset_string.replace("\n", "<br>")
        self.vis.text(dataset_string, opts={"title": "Dataset"})

    def log_implementation(self, params):
        if self.disabled:
            return
        implementation_string = ""
        for param, value in params.items():
            implementation_string += "<b>%s</b>: %s\n" % (param, value)
            implementation_string = implementation_string.replace("\n", "<br>")
        self.implementation_string = implementation_string
        self.implementation_win = self.vis.text(
            implementation_string,
            opts={"title": "Training implementation"}
        )

    def draw_projections(self, embeds, utterances_per_speaker, step, out_fpath=None,
                         max_speakers=10):
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
        self.log_image(out_fpath, step=step)

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