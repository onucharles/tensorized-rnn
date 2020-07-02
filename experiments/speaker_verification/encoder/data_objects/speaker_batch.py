import numpy as np
from typing import List
from encoder.data_objects.speaker import Speaker

class SpeakerBatch:
    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        """
        Training batches contain random slices of size <n_frames> from each utterance.
        :param speakers:
        :param utterances_per_speaker:
        :param n_frames:
        """
        self.speakers = speakers
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}
        
        # Array of shape (n_speakers * n_utterances, n_frames, mel_n), e.g. for 3 speakers with
        # 4 utterances each of 160 frames of 40 mel coefficients: (12, 160, 40)
        self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])

class SpeakerTestBatch:
    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        """
        Test batches contain all utterances from each speaker. Utterances are broken down to
        equal sized frames of <n_frames>, such that each speaker is stored with x <n_frames>-sized frames.
        x could be different for each speaker.
        :param speakers:
        :param n_frames:
        """
        self.speakers = speakers
        self.data = {}      # key: Speaker, Value: (List<int>, List<np.array>)

        for s in speakers:
            utterances = s.get_utterances(utterances_per_speaker)
            frames_batch_per_utterance = []
            n_slices_per_utterance = []
            for utterance in utterances:
                frames = utterance.get_frames()     # n x 40
                mel_slices = compute_partial_slices(len(frames), n_frames, overlap=0.5)
                frames_batch = np.array([frames[s] for s in mel_slices])    # shape: (x, n_frames, n_mel)

                frames_batch_per_utterance.append(frames_batch)
                n_slices_per_utterance.append(len(frames_batch))

            self.data[s] = (frames_batch_per_utterance, n_slices_per_utterance)


def compute_partial_slices(n_frames, partial_utterance_n_frames, overlap=0.5):
    """
    Computes where to split a  mel spectrogram to obtain partial utterances of
    <partial_utterance_n_frames> each.

    :param n_frames: the number of frames in the mel spectogram
    :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial
    utterance
    :param overlap: by how much the partial utterance should overlap. If set to 0, the partial
    utterances are entirely disjoint.
    :return: the mel spectrogram slices as lists of array slices.
    """
    assert 0 <= overlap < 1
    assert n_frames >= partial_utterance_n_frames

    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    mel_slices = []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        mel_slices.append(slice(*mel_range))

    # Always drop last frame if it is not up to <partial_utterance_n_frames>
    last_mel_range = mel_slices[-1]
    coverage = (n_frames - last_mel_range.start) / (last_mel_range.stop - last_mel_range.start)
    if coverage != 1.0:
        mel_slices = mel_slices[:-1]
    return mel_slices
