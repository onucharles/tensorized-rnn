from .lstm import LSTMCell, LSTM
from .tt_linear import TTLinear_lowmem


class TTLSTMCell(LSTMCell):
    def __init__(self, input_size, hidden_size, bias, device, n_cores, tt_rank):
        self.n_cores = n_cores
        self.tt_rank = tt_rank
        super().__init__(input_size, hidden_size, bias, device)

    def _create_input_hidden_weights(self):
        # Uncomment following line for original behavior
        # return TTLinear(in_features=self.input_size, out_features=4 * self.hidden_size,
        return TTLinear_lowmem(in_features=self.input_size, out_features=4 * self.hidden_size,
                        bias=self.bias, auto_shapes=True, d=self.n_cores,
                        tt_rank=self.tt_rank).to(self.device)

    def _create_hidden_hidden_weights(self):
        # Uncomment following line for original behavior
        # return TTLinear(in_features=self.hidden_size, out_features=4 * self.hidden_size,
        return TTLinear_lowmem(in_features=self.hidden_size, out_features=4 * self.hidden_size,
                        bias=self.bias, auto_shapes=True, d=self.n_cores,
                        tt_rank=self.tt_rank).to(self.device)


class TTLSTM(LSTM):
    def __init__(self, input_size, hidden_size, num_layers, device, bias, n_cores, tt_rank):
        self.n_cores = n_cores
        self.tt_rank = tt_rank
        super().__init__(input_size, hidden_size, num_layers, device, bias)

    def _create_first_layer_cell(self):
        return TTLSTMCell(self.input_size, self.hidden_size, self.bias, self.device,
                          n_cores=self.n_cores, tt_rank=self.tt_rank)

    def _create_other_layer_cell(self):
        return TTLSTMCell(self.hidden_size, self.hidden_size, self.bias, self.device,
                          n_cores=self.n_cores, tt_rank=self.tt_rank)
