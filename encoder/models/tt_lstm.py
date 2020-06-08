import torch
from .lstm import LSTMCell, LSTM
from t3nsor.layers import TTLinear


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


### New TTLinear class, hopefully with lower memory usage ###

class TTLinear_lowmem(TTLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        weight_t = self.weight_t
        x_t = x.transpose(0, 1)
        if self.bias is None:
            return tt_expand_matmul(weight_t, x_t).transpose(0, 1)
        else:
            return tt_expand_matmul(weight_t, x_t).transpose(0, 1) + self.bias

def tt_expand_matmul(tt_matrix_a, matrix_b):
    """Multiplies a TT-matrix by a regular matrix, returns a regular matrix.
    Based on t3.tt_dense_matmul
    Args:
    tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
    matrix_b: torch.Tensor of size N x P
    Returns
    torch.Tensor of size M x P
    """
    ndims = tt_matrix_a.ndims
    a_columns = tt_matrix_a.shape[1]
    a_shape = tt_matrix_a.shape
    b_rows = matrix_b.shape[0]
    if a_columns is not None and b_rows is not None:
        if a_columns != b_rows:
            raise ValueError('Arguments shapes should align got %d and %d instead.' %
                       (tt_matrix_a.shape, matrix_b.shape))

    # Iteratively build up dense matrix_a from tt_matrix_a
    matrix_a = torch.ones((1, 1, 1))    # (l_dim, r_dim, tt_rank_dim)
    for core in tt_matrix_a.tt_cores:

        # Merge partially-expanded matrix with new core
        matrix_a = torch.einsum('ija,aklb->ikjlb', matrix_a, core)
        
        # Reshape to give new partially-expanded matrix
        shape = matrix_a.shape
        new_shape = (shape[0]*shape[1], shape[2]*shape[3], shape[4])
        matrix_a = matrix_a.contiguous().view(new_shape)

    assert all((s1 == s2 for s1, s2 in zip(matrix_a.shape, a_shape+[1])))
    matrix_a = matrix_a[:, :, 0]
            
    # After expanding TT weight matrix, multiply with dense matrix_b
    return matrix_a @ matrix_b
