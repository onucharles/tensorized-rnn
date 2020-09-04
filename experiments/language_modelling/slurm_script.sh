#!/bin/bash
# #SBATCH --partition=unkillable
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -o /home/mila/m/millerja/tensorized-rnn/experiments/language_modelling/log-%j.out


# python main.py --enable_logging --epochs 200 --optimizer sgd --lr 5. --train_frac 0.2
# python main.py --enable_logging --epochs 200 --optimizer sgd --lr 5. --train_frac 1.

### Adam low-rank TT
# python main.py --enable_logging --epochs 200 --tt --ttrank 2 --ncores 2 --optimizer adam --lr 0.01 --train_frac 0.2
# python main.py --enable_logging --epochs 200 --tt --ttrank 2 --ncores 3 --optimizer adam --lr 0.01 --train_frac 0.2
# python main.py --enable_logging --epochs 200 --tt --ttrank 2 --ncores 2 --optimizer adam --lr 0.01 --train_frac 1.
# python main.py --enable_logging --epochs 200 --tt --ttrank 2 --ncores 3 --optimizer adam --lr 0.01 --train_frac 1.

# python main.py --enable_logging --epochs 200 --tt --ttrank 4 --ncores 2 --optimizer adam --lr 0.01 --train_frac 0.2
# python main.py --enable_logging --epochs 200 --tt --ttrank 4 --ncores 3 --optimizer adam --lr 0.01 --train_frac 0.2
# python main.py --enable_logging --epochs 200 --tt --ttrank 4 --ncores 2 --optimizer adam --lr 0.01 --train_frac 1.
# python main.py --enable_logging --epochs 200 --tt --ttrank 4 --ncores 3 --optimizer adam --lr 0.01 --train_frac 1.


python main.py --enable_logging --epochs 150 --optimizer sgd --lr 5. --train_frac 0.1 --clip 0.
python main.py --enable_logging --epochs 150 --tt --ttrank 10 --ncores 2 --optimizer sgd --lr 5. --train_frac 0.1 --clip 0.
python main.py --enable_logging --epochs 150 --tt --ttrank 10 --ncores 2 --optimizer sgd --lr 5. --train_frac 0.2 --clip 0.
python main.py --enable_logging --epochs 150 --tt --ttrank 10 --ncores 2 --optimizer sgd --lr 5. --train_frac 0.3 --clip 0.
python main.py --enable_logging --epochs 150 --tt --ttrank 10 --ncores 2 --optimizer sgd --lr 5. --train_frac 0.4 --clip 0.
python main.py --enable_logging --epochs 150 --tt --ttrank 10 --ncores 2 --optimizer sgd --lr 5. --train_frac 0.5 --clip 0.
python main.py --enable_logging --epochs 150 --tt --ttrank 10 --ncores 2 --optimizer sgd --lr 5. --train_frac 0.6 --clip 0.
python main.py --enable_logging --epochs 150 --tt --ttrank 10 --ncores 2 --optimizer sgd --lr 5. --train_frac 0.7 --clip 0.
python main.py --enable_logging --epochs 150 --tt --ttrank 10 --ncores 2 --optimizer sgd --lr 5. --train_frac 0.8 --clip 0.
python main.py --enable_logging --epochs 150 --tt --ttrank 10 --ncores 2 --optimizer sgd --lr 5. --train_frac 0.9 --clip 0.
python main.py --enable_logging --epochs 150 --tt --ttrank 10 --ncores 2 --optimizer sgd --lr 5. --train_frac 1.0 --clip 0.

# python main.py --enable_logging --epochs 150 --optimizer sgd --lr 5. --train_frac 0.2
# python main.py --enable_logging --epochs 150 --optimizer sgd --lr 5. --train_frac 0.3
# python main.py --enable_logging --epochs 150 --optimizer sgd --lr 5. --train_frac 0.4
# python main.py --enable_logging --epochs 150 --optimizer sgd --lr 5. --train_frac 0.5
# python main.py --enable_logging --epochs 150 --optimizer sgd --lr 5. --train_frac 0.6
# python main.py --enable_logging --epochs 150 --optimizer sgd --lr 5. --train_frac 0.7
# python main.py --enable_logging --epochs 150 --optimizer sgd --lr 5. --train_frac 0.8
# python main.py --enable_logging --epochs 150 --optimizer sgd --lr 5. --train_frac 0.9
# python main.py --enable_logging --epochs 150 --optimizer sgd --lr 5. --train_frac 1.


# python main.py --enable_logging --epochs 200 --tt --ttrank 6 --ncores 2 --optimizer adam --lr 0.01 --train_frac 0.2
# python main.py --enable_logging --epochs 200 --tt --ttrank 6 --ncores 3 --optimizer adam --lr 0.01 --train_frac 0.2
# python main.py --enable_logging --epochs 200 --tt --ttrank 6 --ncores 2 --optimizer adam --lr 0.01 --train_frac 1.
# python main.py --enable_logging --epochs 200 --tt --ttrank 6 --ncores 3 --optimizer adam --lr 0.01 --train_frac 1.

# python main.py --enable_logging --epochs 200 --tt --ttrank 8 --ncores 2 --optimizer adam --lr 0.01 --train_frac 0.2
# python main.py --enable_logging --epochs 200 --tt --ttrank 8 --ncores 3 --optimizer adam --lr 0.01 --train_frac 0.2
# python main.py --enable_logging --epochs 200 --tt --ttrank 8 --ncores 2 --optimizer adam --lr 0.01 --train_frac 1.
# python main.py --enable_logging --epochs 200 --tt --ttrank 8 --ncores 3 --optimizer adam --lr 0.01 --train_frac 1.

# python main.py --enable_logging --epochs 200 --tt --ttrank 10 --ncores 2 --optimizer adam --lr 0.01 --train_frac 0.2
# python main.py --enable_logging --epochs 200 --tt --ttrank 10 --ncores 3 --optimizer adam --lr 0.01 --train_frac 0.2
# python main.py --enable_logging --epochs 200 --tt --ttrank 10 --ncores 2 --optimizer adam --lr 0.01 --train_frac 1.
# python main.py --enable_logging --epochs 200 --tt --ttrank 10 --ncores 3 --optimizer adam --lr 0.01 --train_frac 1.


# python main.py --enable_logging --epochs 200 --train_frac 1. --tt --ttrank 2 --ncores 2
# python main.py --enable_logging --epochs 200 --train_frac 1. --tt --ttrank 2 --ncores 3

# python main.py --enable_logging --epochs 200 --train_frac 1. --tt --ttrank 4 --ncores 2
# python main.py --enable_logging --epochs 200 --train_frac 1. --tt --ttrank 4 --ncores 3
# python main.py --enable_logging --epochs 200 --train_frac 1. --tt --ttrank 6 --ncores 3

# python main.py --enable_logging --epochs 200 --tt --ttrank 6 --ncores 2

# python main.py --enable_logging --epochs 100 --train_frac 1. --lr 0.01 --tt --ttrank 4 --ncores 2
# python main.py --enable_logging --epochs 100 --train_frac 1. --lr 0.01 --tt --ttrank 4 --ncores 3
# python main.py --enable_logging --epochs 100 --train_frac 1. --lr 0.01 --tt --ttrank 6 --ncores 2
# python main.py --enable_logging --epochs 100 --train_frac 1. --lr 0.01 --tt --ttrank 6 --ncores 3
# python main.py --enable_logging --epochs 100 --train_frac 1. --lr 0.01 --tt --ttrank 8 --ncores 2
# python main.py --enable_logging --epochs 100 --train_frac 1. --lr 0.01 --tt --ttrank 8 --ncores 3
# python main.py --enable_logging --epochs 100 --train_frac 1. --lr 0.01 --tt --ttrank 10 --ncores 2
# python main.py --enable_logging --epochs 100 --train_frac 1. --lr 0.01 --tt --ttrank 10 --ncores 3
