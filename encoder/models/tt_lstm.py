import torch
from torch import nn


class TT_LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias, device):
        super(TT_LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.input_weights = nn.Linear(input_size, 4 * hidden_size, bias).to(device)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size, bias).to(device)

    def init_hidden(self, batch_size, hidden_size):
        h = torch.zeros(batch_size, hidden_size).to(self.device)
        c = torch.zeros(batch_size, hidden_size).to(self.device)
        return h, c

    def forward(self, input, hx, cx):
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
                #name = 'cell{}'.format(i)
                #lstm_cell = getattr(self, name)
                lstm_cell = self._all_layers[i]

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
