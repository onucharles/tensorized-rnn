
# Model parameters
model_hidden_size = 768
model_embedding_size = 256
model_num_layers = 1

# Training parameters
n_steps = 2e4
learning_rate_init = 2e-3
speakers_per_batch = 16
utterances_per_speaker = 32

## Tensor-train parameters for last linear layer.
compression = None
n_cores = 0
rank = 0

# Evaluation and Test parameters
val_speakers_per_batch = 40
val_utterances_per_speaker = 32
test_speakers_per_batch = 40
test_utterances_per_speaker = 32

# seed = None
