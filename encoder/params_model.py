
# Model parameters
model_hidden_size = 512
model_embedding_size = 256
model_num_layers = 1

# Training parameters
n_steps = 2e4
learning_rate_init = 1e-3
speakers_per_batch = 64
utterances_per_speaker = 32

## Tensor-train parameters for last linear layer.
use_tt = False
n_cores = 2
tt_rank = 2


# Evaluation and Test parameters
val_speakers_per_batch = 40
val_utterances_per_speaker = 32
test_speakers_per_batch = 40
test_utterances_per_speaker = 32
