import torch
import torch.nn.functional as F
from torch import nn

from context import tensorized_rnn
from tensorized_rnn.lstm import LSTM
from t3nsor.layers import TTLinear
from tensorized_rnn.tt_lstm import TTLSTM


class MNIST_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(MNIST_Classifier, self).__init__()

        ### PYTORCH LSTM
        # self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            # num_layers=num_layers, batch_first=True)
        # self.linear = nn.Linear(hidden_size, output_size)

        ### CUSTOM LSTM
        device = torch.device('cuda:1')
        self.lstm = TTLSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=num_layers, device=device, bias=True,
                             n_cores=2, tt_rank=2)
        self.linear = TTLinear(in_features=hidden_size, out_features=output_size,
                               bias=True, auto_shapes=True, d=2, tt_rank=2).to(device)

    def forward(self, inputs):
        out, (last_hidden, last_cell) = self.lstm(inputs)

        ### PYTORCH LSTM
        # o = self.linear(out[:, -1, :])

        ### CUSTOM LSTM
        o = self.linear(last_hidden)

        return F.log_softmax(o, dim=1)
