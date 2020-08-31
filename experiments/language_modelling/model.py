import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from context import tensorized_rnn
from tensorized_rnn.lstm import LSTM
from tensorized_rnn.tt_lstm import TTLSTM
from tensorized_rnn.gru import GRU, TTGRU
from t3nsor.layers import TTLinear, TTEmbedding
from tensorized_rnn.grad_tools import param_count as pc

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, device, dropout=0.5, tie_weights=False,
                 tt=False, n_cores=3, tt_rank=2, naive_tt=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.tt = tt
        self.drop = nn.Dropout(dropout)

        if tt:
            self.encoder = TTEmbedding(voc_size=ntoken, emb_size=ninp, auto_shapes=True, d=n_cores,
                                       tt_rank=tt_rank)
            if rnn_type == 'LSTM':
                self.rnn = TTLSTM(input_size=ninp, hidden_size=nhid, num_layers=nlayers, device=device,
                                  n_cores=n_cores, tt_rank=tt_rank, log_grads=False, is_naive=naive_tt)
            elif rnn_type == 'GRU':
                if naive_tt: raise ValueError("Naive TT is not implemented for GRU.")
                self.rnn = TTGRU(input_size=ninp, hidden_size=nhid, num_layers=nlayers, device=device,
                                 n_cores=n_cores, tt_rank=tt_rank, log_grads=False)
            else:
                raise ValueError("Unknown TT-RNN type was passed: ", rnn_type)

            self.decoder = TTLinear(in_features=nhid, out_features=ntoken, auto_shapes=True, d=n_cores,
                                    tt_rank=tt_rank)
        else:
            self.encoder = nn.Embedding(ntoken, ninp)
            if rnn_type == 'LSTM':
                self.rnn = LSTM(input_size=ninp, hidden_size=nhid, num_layers=nlayers, device=device)
            elif rnn_type == 'GRU':
                self.rnn = GRU(input_size=ninp, hidden_size=nhid, num_layers=nlayers, device=device)
            else:
                raise ValueError("Unknown TT-RNN type was passed.")

            self.decoder = nn.Linear(nhid, ntoken)

            ### Using PyTorch LSTM and GRU
            # self.encoder = nn.Embedding(ntoken, ninp)

            # if rnn_type in ['LSTM', 'GRU']:
                # self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
            # else:
                # try:
                    # nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                # except KeyError:
                    # raise ValueError( """An invalid option for `--model` was supplied,
                                     # options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
                # self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout, batch_first=True)
            # self.decoder = nn.Linear(nhid, ntoken)


        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.tie_weights = tie_weights

    def init_weights(self):
        if self.tt:
            return
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        # input dim: seq_len, batch_size

        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb.transpose(1, 0), hidden) 
        output = self.drop(output.transpose(1, 0))
        output = output.reshape(-1, self.nhid)
        decoded = self.decoder(output)
        # decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def param_count(self):
        if self.tie_weights:
            return pc(self.encoder) + pc(self.rnn)
        else:
            return pc(self.encoder) + pc(self.rnn) + pc(self.decoder)
