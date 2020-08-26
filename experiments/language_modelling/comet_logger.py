import pickle
import warnings

from comet_ml import Experiment, ExistingExperiment

# from config_j import COMET_API_KEY, COMET_WORKSPACE, PROJECT_NAME
from config import COMET_API_KEY, COMET_WORKSPACE, PROJECT_NAME

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

    def log_metric(self, *args, **kwargs):
        self.experiment.log_metric(*args, **kwargs)

    def log_metrics(self, *args, **kwargs):
        self.experiment.log_metrics(*args, **kwargs)

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

