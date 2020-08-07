#!/bin/sh
# Experimental configs to run
# python pmnist_test.py --cuda --log_grads --enable_logging --gru \
#                       --tt --ncores 4 --ttrank 5

python pmnist_test.py --gru --cuda --log_grads --enable_logging \
                      --tt --ncores 2 --ttrank 5
# python pmnist_test.py --gru --cuda --log_grads --enable_logging \
#                       --tt --ncores 3 --ttrank 5

# python pmnist_test.py --cuda --log_grads --enable_logging \
#                       --tt --ncores 2 --ttrank 5
# python pmnist_test.py --cuda --log_grads --enable_logging \
#                       --tt --ncores 3 --ttrank 5

# python pmnist_test.py --cuda --log_grads \
#                       --tt --ncores 4 --ttrank 5