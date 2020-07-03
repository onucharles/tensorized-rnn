import torch
import torch.nn.functional as F
from torch import nn

from context import tensorized_rnn
from tensorized_rnn.lstm import LSTM
from t3nsor.layers import TTLinear
from tensorized_rnn.tt_lstm import TTLSTM


class MNIST_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, device,
                 tt_lstm=True, n_cores=3, tt_rank=2):
        super(MNIST_Classifier, self).__init__()
        self.tt_lstm = tt_lstm

        if tt_lstm:
            ### TT LSTM
            # device = (torch.device('cuda:0') if torch.cuda.is_available()
            #           else 'cpu')
            self.lstm = TTLSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, device=device,
                                n_cores=n_cores, tt_rank=tt_rank)
            self.linear = TTLinear(in_features=hidden_size, out_features=output_size,
                                   bias=True, auto_shapes=True, d=n_cores, tt_rank=tt_rank).to(device)
        else:
            ### REGULAR LSTM
            self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, device=device)
            self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        out, (last_hidden, last_cell) = self.lstm(inputs)
        o = self.linear(last_hidden)
        #o = self.linear(out[:, -1, :])

        return F.log_softmax(o, dim=1)
