from utils.argutils import print_args
from encoder.main import test
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tests a trained encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-d", "--clean_data_root", type=Path, help= \
        "Path to the output directory of encoder_preprocess.py for training, validation and test sets.",
                        default=r"..\..\..\_experiments\speech-model-compression\speaker-verification")
    parser.add_argument("-m", "--exp_root_dir", type=Path, default="", help="Root directory for project's experiments")
    parser.add_argument("-k", "--prev_exp_key", type=str, default=None, help= \
        "The comet key of experiment to whose model is being tested.")
    parser.add_argument("--enable_comet", action="store_true", help= \
        "Enable logging to comet or file system.")
    parser.add_argument("--gpu_no", type=int, default=0, help =\
        "The index of GPU to use if multiple are available. If none, CPU will be used.")
    parser.add_argument('--use_gru', action='store_true',
                        help='Use GRU instead of LSTM')
    args = parser.parse_args()

    # Run the test
    print_args(args, parser)
    test(**vars(args))

