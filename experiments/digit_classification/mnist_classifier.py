import torch
import torch.nn.functional as F
from torch import nn

from context import tensorized_rnn
from tensorized_rnn.lstm import LSTM
from tensorized_rnn.tt_lstm import TTLSTM
from tensorized_rnn.gru import GRU, TTGRU
from tensorized_rnn.rnn_utils import param_count as pc
from t3nsor.layers import TTLinear


class MNIST_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, device,
                 tt=True, gru=True, n_cores=3, tt_rank=2, log_grads=False, 
                 naive_tt=False):
        super(MNIST_Classifier, self).__init__()
        self.gru = gru

        if tt:
            if not gru:
                self.rnn = TTLSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, device=device,
                                n_cores=n_cores, tt_rank=tt_rank, 
                                log_grads=log_grads, is_naive=naive_tt)
            else:
                self.rnn = TTGRU(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, device=device,
                                n_cores=n_cores, tt_rank=tt_rank, 
                                log_grads=log_grads, is_naive=naive_tt)
            self.linear = TTLinear(in_features=hidden_size, 
                    out_features=output_size, bias=True, 
                    auto_shapes=True, d=n_cores, tt_rank=tt_rank)
        else:
            if not gru:
                self.rnn = LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, device=device, 
                                log_grads=log_grads)
            else:
                self.rnn = GRU(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, device=device, 
                                log_grads=log_grads)
            self.linear = nn.Linear(hidden_size, output_size)

    def param_count(self):
        return self.rnn.param_count() + pc(self.linear)

    def forward(self, inputs):
        if self.gru:
            out, last_hidden = self.rnn(inputs)
        else:
            out, (last_hidden, last_cell) = self.rnn(inputs)
        o = self.linear(out[:, -1, :])

        return F.log_softmax(o, dim=1)
