from pathlib import Path
import numpy as np

dataset_root = Path("/home/ml/c.onu/experiments/speech-model-compression/speaker-verification/_librispeech_dev-clean_tisv")
nans = []
zeros = []
infs = []
for dir in dataset_root.glob("*"):
    if not dir.is_dir(): continue
    
    with dir.joinpath("_sources.txt").open("r") as sources_file:
        sources = [l.split(",") for l in sources_file]
   
    for frames_fname, wave_fpath in sources:
        frames_fpath = dir.joinpath(frames_fname)
        cur_frames = np.load(frames_fpath)
        n_nans = np.sum(np.isnan(cur_frames))
        n_zeros = np.sum(cur_frames == 0)
        n_infs = np.sum(np.isinf(cur_frames))
        print("Frame size is: {}\t No of nans is: {}\t No of zeros: {}"
                .format(cur_frames.shape, n_nans, n_zeros))
        if n_nans > 0: nans.append(frames_fname)
        if n_zeros > 0: zeros.append(frames_fname)
        if n_infs > 0: infs.append(frames_fname)


print("Nans culprits: ", nans)
print("Zeros culprits: ", zeros)
print("Infs culprits: ", infs)


