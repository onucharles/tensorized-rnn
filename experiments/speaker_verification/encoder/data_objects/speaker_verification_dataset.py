from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.speaker_batch import SpeakerBatch, SpeakerTestBatch
from encoder.data_objects.speaker import Speaker
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed


class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets_root, dataset_len, train_frac):
        """
        :param datasets_root:
        :param dataset_len: the number of pseudo-speakers in the dataset. Speakers are sampled
                            with replacement. Each time a speaker is returned, a random set of
                            utterances and random segment from each utterance is selected.
        :param train_frac: the fraction of training set to use.
        """
        self.root = datasets_root
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        n_speakers = len(speaker_dirs)
        if n_speakers == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")

        n_speakers_to_use = int(train_frac * n_speakers)
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs[:n_speakers_to_use]]

        self.speaker_cycler = RandomCycler(self.speakers)
        self.dataset_len = dataset_len

        print("Training set - number of speakers is {} ({}% of total)".format(len(self.speakers), train_frac * 100))

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
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, partials_n_frames,
                 sampler=None, batch_sampler=None, num_workers=0, pin_memory=False, timeout=0,
                 worker_init_fn=None, drop_last=False):
        self.utterances_per_speaker = utterances_per_speaker
        self.partials_n_frames = partials_n_frames

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
        return SpeakerBatch(speakers, self.utterances_per_speaker, self.partials_n_frames)

class SpeakerVerificationTestDataLoader(DataLoader):
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, partials_n_frames,
                 sampler=None, batch_sampler=None, num_workers=0, pin_memory=False, timeout=0,
                 worker_init_fn=None, drop_last=False):
        self.utterances_per_speaker = utterances_per_speaker
        self.partials_n_frames = partials_n_frames

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
        return SpeakerTestBatch(speakers, self.utterances_per_speaker, self.partials_n_frames)
