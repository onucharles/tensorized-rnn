import torch
from torch import nn


class TT_LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias, device):
        super(TT_LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        # self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        # self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        # self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        # self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.input_weights = nn.Linear(input_size, 4 * hidden_size, bias).to(device)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size, bias).to(device)
        self.init_weight_and_bias()

    def init_weight_and_bias(self):
        sqrt_k = (1.0 / self.hidden_size) ** .5
        nn.init.uniform_(self.input_weights.weight, -sqrt_k, sqrt_k)
        nn.init.uniform_(self.input_weights.bias, -sqrt_k, sqrt_k)

    def init_hidden(self, batch_size, hidden_size):
        h = torch.zeros(batch_size, hidden_size).to(self.device)
        c = torch.zeros(batch_size, hidden_size).to(self.device)
        return h, c

    def forward(self, input, hx, cx):
        # gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
        #          torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        gates = self.input_weights(input) + self.hidden_weights(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy


class TT_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, bias=True):
        super(TT_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self._all_layers = []

        # instantiate lstm cell for each layer.
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            if i == 0:
                cell = TT_LSTMCell(self.input_size, self.hidden_size, self.bias, device)
            else:
                cell = TT_LSTMCell(self.hidden_size, self.hidden_size, self.bias, device)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        # input: (batch_size, n_steps, input_size)

        batch_size, n_steps, input_size = input.size()
        internal_state = []
        outputs = []

        for step in range(n_steps):
            x = input[:, step, :]
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                lstm_cell = getattr(self, name)

                # initialise each cell at first step.
                if step == 0:
                    (h, c) = lstm_cell.init_hidden(batch_size, self.hidden_size)
                    internal_state.append((h, c))

                (h, c) = internal_state[i]
                x, new_c = lstm_cell(x, h, c)
                internal_state[i] = (x, new_c)

            outputs.append(x)
        outputs = torch.stack(outputs, dim=0)
        return outputs, (x, new_c)
