
# Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3

# Training parameters
n_epochs = 1000
learning_rate_init = 1e-4
speakers_per_batch = 32
utterances_per_speaker = 10

## Tensor-train parameters for last linear layer.
use_tt = True
n_cores = 4
tt_rank = 3

# Evaluation and Test parameters
test_speakers_per_batch = 10
test_utterances_per_speaker = 50
test_n_epochs = 5
