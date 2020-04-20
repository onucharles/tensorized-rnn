from utils.argutils import print_args
from encoder.test import test
from pathlib import Path
import argparse

import torch
import random
import numpy as np

def set_seed(seed=None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tests a trained encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # parser.add_argument("-r", "--run_id", default="someid", type=str, help="Experiment ID")
    parser.add_argument("-d", "--test_data_dir", type=Path, help="Path to the test data directory (processed).")
    parser.add_argument("-m", "--model_path", type=Path, default="", help="Path to saved model to use.")
    parser.add_argument("-w", "--n_workers", type=int, default=1,
                        help="Number of processor cores to use if CUDA is not available")
    # parser.add_argument("--visdom_server", type=str, default="http://localhost")
    # parser.add_argument("--no_visdom", action="store_true", help= "Disable visdom.")
    args = parser.parse_args()

    set_seed(11)

    # Run the training
    print_args(args, parser)
    test(**vars(args))

