from utils.argutils import print_args
from encoder.main import test
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tests a trained encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-d", "--test_data_dir", type=Path, help="Path to the test data directory (processed).")
    parser.add_argument("-m", "--exp_root_dir", type=Path, default="", help="Root directory for project's experiments")
    parser.add_argument("-k", "--prev_exp_key", type=str, default=None, help= \
        "The comet key of experiment to whose model is being tested.")
    parser.add_argument("--no_comet", action="store_true", help= \
        "Disable comet logging.")
    parser.add_argument("--gpu_no", type=int, default=0, help =\
        "The index of GPU to use if multiple are available. If none, CPU will be used.")
    args = parser.parse_args()

    # Run the test
    print_args(args, parser)
    test(**vars(args))

