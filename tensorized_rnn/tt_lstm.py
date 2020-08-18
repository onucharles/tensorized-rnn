from t3nsor.layers import TTLinear
from .lstm import LSTMCell, LSTM
from .tt_linearset import TTLinearSet
from .rnn_utils import tt_shape

class TTLSTMCell(LSTMCell):
    def __init__(self, input_size, hidden_size, bias, device, n_cores, tt_rank, 
                 is_naive=False, new_core=None):
        self.n_cores = n_cores
        self.tt_rank = tt_rank
        self.is_naive = is_naive
        self.new_core = new_core
        self.n_gate = 4
        super().__init__(input_size, hidden_size, bias, device)

    def _create_input_hidden_weights(self):
        if self.is_naive:
            return TTLinearSet(in_features=self.input_size,
                            out_features=self.hidden_size, n_gates=self.n_gate,
                            bias=False, auto_shapes=True,
                            d=self.n_cores, tt_rank=self.tt_rank).to(self.device)
        else:
            shape = tt_shape(self.input_size, self.hidden_size, 
                             self.n_cores, self.n_gate, new_core=self.new_core)
            return TTLinear(out_features=self.n_gate*self.hidden_size, shape=shape, 
                            bias=self.bias, auto_shapes=False, d=self.n_cores, 
                            tt_rank=self.tt_rank).to(self.device)

    def _create_hidden_hidden_weights(self):
        if self.is_naive:
            return TTLinearSet(in_features=self.hidden_size,
                            out_features=self.hidden_size, n_gates=self.n_gate,
                            bias=False, auto_shapes=True,
                            d=self.n_cores, tt_rank=self.tt_rank).to(self.device)
        else:
            shape = tt_shape(self.hidden_size, self.hidden_size, 
                             self.n_cores, self.n_gate, new_core=self.new_core)
            return TTLinear(out_features=self.n_gate*self.hidden_size, shape=shape, 
                            bias=self.bias, auto_shapes=False, d=self.n_cores, 
                            tt_rank=self.tt_rank).to(self.device)


class TTLSTM(LSTM):
    def __init__(self, input_size, hidden_size, num_layers, device, n_cores, 
                 tt_rank, bias=True, is_naive=False, log_grads=False, new_core=None):
        assert new_core in [None, 'first', 'last']
        self.n_cores = n_cores
        self.tt_rank = tt_rank
        self.is_naive = is_naive
        self.new_core = new_core
        super().__init__(input_size, hidden_size, num_layers, device, bias,
                         log_grads=log_grads)

    def _create_first_layer_cell(self):
        return TTLSTMCell(self.input_size, self.hidden_size, self.bias, self.device,
                          n_cores=self.n_cores, tt_rank=self.tt_rank, 
                          is_naive=self.is_naive, new_core=self.new_core)

    def _create_other_layer_cell(self):
        return TTLSTMCell(self.hidden_size, self.hidden_size, self.bias, self.device,
                          n_cores=self.n_cores, tt_rank=self.tt_rank, 
                          is_naive=self.is_naive, new_core=self.new_core)
