import torch
from t3nsor.layers import TTLinear

class TTLinear_lowmem(TTLinear):
    """ TTLinear class with lower memory usage """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # do forward pass on CPU to save GPU memory.
        cpu_device = torch.device('cpu')

        weight_t = self.weight_t.to(cpu_device)
        x_t = x.transpose(0, 1).to(cpu_device)

        if self.bias is None:
            return tt_expand_matmul(weight_t, x_t, cpu_device).transpose(0, 1).to(x.device)
        else:
            return tt_expand_matmul(weight_t, x_t, cpu_device).transpose(0, 1).to(x.device) + self.bias


def tt_expand_matmul(tt_matrix_a, matrix_b, device):
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
    matrix_a = torch.ones((1, 1, 1)).to(device)  # (l_dim, r_dim, tt_rank_dim)
    for core in tt_matrix_a.tt_cores:
        # Merge partially-expanded matrix with new core
        matrix_a = torch.einsum('ija,aklb->ikjlb', matrix_a, core)

        # Reshape to give new partially-expanded matrix
        shape = matrix_a.shape
        new_shape = (shape[0] * shape[1], shape[2] * shape[3], shape[4])
        matrix_a = matrix_a.contiguous().view(new_shape)

    assert all((s1 == s2 for s1, s2 in zip(matrix_a.shape, a_shape + [1])))
    matrix_a = matrix_a[:, :, 0]

    # After expanding TT weight matrix, multiply with dense matrix_b
    return matrix_a @ matrix_b
