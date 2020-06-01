from .lstm import LSTMCell, LSTM
from t3nsor.layers import TTLinear

class TTLSTMCell(LSTMCell):
    def __init__(self, input_size, hidden_size, bias, device, n_cores, tt_rank):
        super().__init__(input_size, hidden_size, bias, device)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.input_weights = TTLinear(in_features=input_size, out_features=4 * hidden_size,
                            bias=bias, auto_shapes=True, d=n_cores, tt_rank=tt_rank).to(device)
        self.hidden_weights = TTLinear(in_features=hidden_size, out_features=4 * hidden_size,
                            bias=bias, auto_shapes=True, d=n_cores, tt_rank=tt_rank).to(device)


class TTLSTM(LSTM):
    def __init__(self, input_size, hidden_size, num_layers, device, bias, n_cores, tt_rank):
        super().__init__(input_size, hidden_size, num_layers, device, bias)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        # instantiate lstm cell for each layer.
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            if i == 0:
                cell = LSTMCell(self.input_size, self.hidden_size, self.bias, device,
                                n_cores=n_cores, tt_rank=tt_rank)
            else:
                cell = LSTMCell(self.hidden_size, self.hidden_size, self.bias, device,
                                n_cores=n_cores, tt_rank=tt_rank)
            setattr(self, name, cell)
            self._all_layers.append(cell)
