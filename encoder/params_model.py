
# Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3

# Training parameters
n_steps = 5e4
learning_rate_init = 1e-3
speakers_per_batch = 64
utterances_per_speaker = 32

## Tensor-train parameters for last linear layer.
use_tt = True
n_cores = 2
tt_rank = 3

# Evaluation and Test parameters
test_speakers_per_batch = 10 #32
test_utterances_per_speaker = 10 #50
# test_n_epochs = 5
