import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias, device):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.input_weights = self._create_input_hidden_weights()
        self.hidden_weights = self._create_hidden_hidden_weights()

    def _create_input_hidden_weights(self):
        return nn.Linear(self.input_size, 4 * self.hidden_size, self.bias).to(self.device)

    def _create_hidden_hidden_weights(self):
        return nn.Linear(self.hidden_size, 4 * self.hidden_size, self.bias).to(self.device)

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


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, bias=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device

        # instantiate lstm cell for each layer.
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            if i == 0:
                cell = self._create_first_layer_cell()
            else:
                cell = self._create_other_layer_cell()
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def _create_first_layer_cell(self):
        return LSTMCell(self.input_size, self.hidden_size, self.bias, self.device)

    def _create_other_layer_cell(self):
        return LSTMCell(self.hidden_size, self.hidden_size, self.bias, self.device)

    def init_hidden(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return h, c

    def forward(self, input, init_states=None):
        # TODO: Clearly document input and output shapes.
        # input: (batch_size, n_steps, input_size)

        batch_size, n_steps, input_size = input.size()
        outputs = []    # TODO : Change to torch array to make output identical to torch.nn.LSTM

        (h, c) = self.init_hidden(batch_size) if init_states is None else init_states
        internal_state = [(h, c)] * self.num_layers

        for step in range(n_steps):
            x = input[:, step, :]
            for i in range(self.num_layers):
                # name = 'cell{}'.format(i)
                # lstm_cell = getattr(self, name)
                lstm_cell = self._all_layers[i]

                (h, c) = internal_state[i]
                x, new_c = lstm_cell(x, h, c)
                internal_state[i] = (x, new_c)

            outputs.append(x)

        # outputs = torch.stack(outputs, dim=0)
        return outputs, (x, new_c)
