from utils.argutils import print_args
from encoder.main import train
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains the speaker encoder. You must have run encoder_preprocess.py first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-d", "--clean_data_root", type=Path, help= \
        "Path to the output directory of encoder_preprocess.py for training, validation and test sets.",
        default=r"..\..\..\_experiments\speech-model-compression\speaker-verification" )
    parser.add_argument("-m", "--models_dir", type=Path, help=\
        "Path to the output directory that will contain the saved model weights, as well as "
        "backups of those weights and plots generated during training.",
        default=r"..\..\..\_experiments\speech-model-compression\speaker-verification")
    parser.add_argument("-v", "--val_every", type=int, default=10, help= \
        "Number of steps between updates of the loss and the plots.")
    parser.add_argument("-u", "--umap_every", type=int, default=20, help= \
        "Number of steps between updates of the umap projection. Set to 0 to never update the "
        "projections.")
    parser.add_argument("-r", "--resume_experiment", action="store_true", help= \
        "Resume a saved experiment.")
    parser.add_argument("-k", "--prev_exp_key", type=str, default=None, help= \
        "The comet key of experiment to resume.")
    parser.add_argument("--enable_comet", action="store_true", help= \
        "Enable logging to comet or file system.")
    parser.add_argument("--gpu_no", type=int, default=0, help =\
        "The index of GPU to use if multiple are available. If none, CPU will be used.")
    parser.add_argument("--seed", type=int, default=11, help= \
        "The random seed to use.")
    parser.add_argument("--train_frac", type=float, default=1.0, help="Fraction of training data to use.")
    parser.add_argument("--log_grad", action="store_true", help="Log gradients.")
    parser.add_argument("--clip", type=int, default=3, help="The max norm for gradient clipping.")

    args = parser.parse_args()

    assert 0.0 < args.train_frac <= 1.0
    
    # Process the arguments
    args.models_dir.mkdir(exist_ok=True)
    
    # Run the training
    print_args(args, parser)
    train(**vars(args))
