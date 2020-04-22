from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.speaker_batch import SpeakerBatch
from encoder.data_objects.speaker import Speaker
from encoder.params_data import partials_n_frames
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets_root, n_epochs):
        """
        :param datasets_root:
        :param n_epochs:
        """
        # :param dataset_len: Can be larger than the number of speakers since speakers will be
        #         sampled with (constrained) replacement. If None, it is set to the no of speakers.
        self.root = datasets_root
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        n_speakers = len(speaker_dirs)
        if n_speakers == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]

        self.speaker_cycler = RandomCycler(self.speakers)
        self.dataset_len = n_epochs * n_speakers
        # self.dataset_len = dataset_len if dataset_len is not None else len(speaker_dirs)
        print("Train dataset length is: {}, ie no_of_epochs({}) x no_of_speakers({})".
              format(self.dataset_len, n_epochs, n_speakers))

    def __len__(self):
        return int(self.dataset_len)

    def __getitem__(self, index):
        return next(self.speaker_cycler)
    
    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string

class SpeakerVerificationTestSet(Dataset):
    def __init__(self, datasets_root):
        """

        :param datasets_root:
        """
        self.root = datasets_root
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        if len(speaker_dirs) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
        self.dataset_len = len(self.speakers)
        print("Test dataset length is: {}".format(self.dataset_len))

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.speakers[item]

    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string
    
class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None, drop_last=False):
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=drop_last,
            timeout=timeout, 
            worker_init_fn=worker_init_fn,
        )

    def collate(self, speakers):
        # TODO: if test batch should contain full length utterances.
        return SpeakerBatch(speakers, self.utterances_per_speaker, partials_n_frames) 
    