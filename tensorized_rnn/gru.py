import torch
from torch import nn
from t3nsor.layers import TTLinear

from .grad_tools import ActivGradLogger


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias, device):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.input_weights = self._create_input_hidden_weights()
        self.hidden_weights = self._create_hidden_hidden_weights()

    def _create_input_hidden_weights(self):
        return nn.Linear(self.input_size, 3 * self.hidden_size, self.bias).to(self.device)

    def _create_hidden_hidden_weights(self):
        return nn.Linear(self.hidden_size, 3 * self.hidden_size, self.bias).to(self.device)

    def forward(self, input, hx):
        # Updates hidden state using the following rules:
        # r_t = \sigma(W_{ir} x_t + b_{ir} +        W_{hr} h_{(t-1)} + b_{hr})  RESET
        # z_t = \sigma(W_{iz} x_t + b_{iz} +        W_{hz} h_{(t-1)} + b_{hz})  UPDATE
        # n_t = \tanh( W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)} + b_{hn})) NEW
        # h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}

        # Apply the stacked weights
        input_part  = self.input_weights(input)
        hidden_part = self.hidden_weights(hx)

        # Update hidden state using the above rules
        hsize = self.hidden_size
        reset_gate  = torch.sigmoid(input_part[:, :hsize] + 
                                    hidden_part[:, :hsize])
        update_gate = torch.sigmoid(input_part[:, hsize:2*hsize] + 
                                    hidden_part[:, hsize:2*hsize])
        new_gate    = torch.tanh(input_part[:, 2*hsize:] + 
                                 reset_gate * hidden_part[:, 2*hsize:])
        hy          = (1 - reset_gate) * new_gate + reset_gate * hx

        # Register gradient hooks if we have them
        if hasattr(self, '_h_backward_hook') and hy.requires_grad:
            hy.register_hook(self._h_backward_hook)

        return hy


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, 
                 bias=True, log_grads=False):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.log_grads = log_grads

        # instantiate gru cell for each layer.
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            if i == 0:
                cell = self._create_first_layer_cell()
            else:
                cell = self._create_other_layer_cell()
            setattr(self, name, cell)
            self._all_layers.append(cell)

        # Add logging for hidden state
        if log_grads:
            for i, cell in enumerate(self._all_layers):
                # Hidden and cell state loggers
                h_logger = ActivGradLogger(f"hidden_{i}")

                # Set up hooks for the loggers. Backward hooks on modules
                # don't work well, so using the Tensor-level backward hook
                h_forward, h_backward = h_logger.create_hooks(0)
                cell.register_forward_hook(h_forward)
                cell._h_backward_hook = h_backward

    def _create_first_layer_cell(self):
        return GRUCell(self.input_size, self.hidden_size, self.bias, self.device)

    def _create_other_layer_cell(self):
        return GRUCell(self.hidden_size, self.hidden_size, self.bias, self.device)

    def init_hidden(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return h

    def forward(self, input, init_states=None):
        """
        :param input:       Tensor of input data of shape (batch_size, seq_len, input_size).
        :param init_states: Initial hidden states of GRU. If None, is initialized to zeros.
                            Shape is (batch_size, hidden_size).

        :return:    outputs, h
                    outputs:  Torch tensor of shape (seq_len, batch_size, hidden_size) 
                              containing output features from last layer of GRU.
                    h:        Output features (ie hiddens state) from last time step of 
                              the last layer. Shape is (batch_size, hidden_size)
        """

        batch_size, seq_len, input_size = input.size()
        outputs = torch.zeros(batch_size, seq_len, self.hidden_size).to(input.device)

        # initialise hidden and cell states.
        h = self.init_hidden(batch_size) if init_states is None else init_states
        internal_state = [h] * self.num_layers

        for step in range(seq_len):
            x = input[:, step, :]
            for i in range(self.num_layers):
                # name = 'cell{}'.format(i)
                # gru_cell = getattr(self, name)
                gru_cell = self._all_layers[i]

                h = internal_state[i]
                x = gru_cell(x, h)
                internal_state[i] = x
            outputs[:, step, :] = x

        return outputs, x


class TTGRUCell(GRUCell):
    def __init__(self, input_size, hidden_size, bias, device, n_cores, tt_rank):
        self.n_cores = n_cores
        self.tt_rank = tt_rank
        super().__init__(input_size, hidden_size, bias, device)

    def _create_input_hidden_weights(self):
        return TTLinear(in_features=self.input_size, 
                        out_features=3*self.hidden_size,
                        bias=self.bias, auto_shapes=True, 
                        d=self.n_cores, tt_rank=self.tt_rank).to(self.device)

    def _create_hidden_hidden_weights(self):
        return TTLinear(in_features=self.hidden_size, 
                        out_features=3*self.hidden_size,
                        bias=self.bias, auto_shapes=True, 
                        d=self.n_cores, tt_rank=self.tt_rank).to(self.device)


class TTGRU(GRU):
    def __init__(self, input_size, hidden_size, num_layers, device, n_cores, 
                 tt_rank, bias=True, log_grads=False):
        self.n_cores = n_cores
        self.tt_rank = tt_rank
        super().__init__(input_size, hidden_size, num_layers, device, bias,
                         log_grads=log_grads)

    def _create_first_layer_cell(self):
        return TTGRUCell(self.input_size, self.hidden_size, self.bias, self.device,
                          n_cores=self.n_cores, tt_rank=self.tt_rank)

    def _create_other_layer_cell(self):
        return TTGRUCell(self.hidden_size, self.hidden_size, self.bias, self.device,
                          n_cores=self.n_cores, tt_rank=self.tt_rank)