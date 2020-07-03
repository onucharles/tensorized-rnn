import torch

from t3nsor.tensor_train import TensorTrainBatch
"""
## NOTE ON TT FORMATTING

Here is the format of the tensor lists comprising a TensorTrainBatch object:

TT objects are lists of Pytorch tensors, where each tensor has the shape:
[batch, left_rank, left_mat, right_mat, right_rank], with the rank indices 
parameterizing the TT ranks and the mat indices parameterizing the
matrix dimensions. The batch index varies over different instances of
the TT matrices, and is identical in all tensor cores of the TT mat.
The left/right global matrix dimensions are the product of the left/right
local matrix dimensions.
"""

def project_ttgrad(base_tt, grad_tt):
    """
    Projects local gradients from cores of TT matrix to dense matrix gradient

    Args:
        base_tt:  Batch core object giving the base TT matrix at which
                  the gradient is evaluated
        grad_tt:  Batch core object giving the corresponding tangent
                  cores for the TT matrix

    Returns:
        grad_mat: Batch of matrices of shape [batch, left_mat, right_mat],
                  where the mat indices parameterize the global tangent
                  matrix
    """
    # Unpack inputs to get core lists
    assert all(isinstance(c, TensorTrainBatch) for c in [base_tt, grad_tt])
    base_cores, grad_cores = base_tt.tt_cores, grad_tt.tt_cores

    # Check formatting of tensors contained in base_cores and grad_cores
    assert len(base_cores) == len(grad_cores)
    assert all(b.shape == g.shape for b, g in zip(base_cores, grad_cores))
    assert all(len(c.shape) == 5 for c in base_cores)
    num_cores = len(base_cores)
    batch_size = base_cores[0].shape[0]
    rank_dims = [1] + [b.shape[-1] for b in base_cores]
    assert all(b.shape[0] == batch_size for b in base_cores)

    # Create a TT matrix representing the global gradient, which has a
    # block structure of the form    ->    [[base_core, grad_core],
    #                                                0, base_core]]
    local_tt = []
    for b_core, g_core in zip(base_cores, grad_cores):
        b, l, lm, rm, r = b_core.shape
        # Template for the TT core of local gradient
        local_core = torch.zeros(b, 2*l, lm, rm, 2*r)
        # Diagonal base core terms
        for i in range(2):
            local_core[:, i*l:(i+1)*l, :, :, i*r:(i+1)*r] = b_core
        # Upper right gradient term
        local_core[:, :l, :, :, r:] = g_core
        # Add core to the TT matrix core list
        local_tt.append(local_core)

    # Pick out the correct boundary conditions to get just the gradient
    local_tt[0]  = local_tt[0][:, 0:1, :, :, :]
    local_tt[-1] = local_tt[-1][:,  :, :, :, 1:2]

    # Convert to TensorTrainBatch and use full() method to get global mats
    return TensorTrainBatch(local_tt).full()

## Below gives an example usage of project_ttgrad
if __name__ == '__main__':
    # Batch size of 10, three cores with TT ranks 5 and 7, 
    # all three core mat dims are 2, which gives an 8x8 global mat
    shapes = [[10, 1, 2, 2, 5], [10, 5, 2, 2, 7], [10, 7, 2, 2, 1]]
    # Cores are chosen as all-ones tensors
    base_cores = grad_cores = [torch.ones(*s) for s in shapes]
    base_tt = TensorTrainBatch(base_cores)
    grad_tt = TensorTrainBatch(grad_cores)

    # Project gradient TT cores at base TT matrix to get global gradients
    grad_mats = project_ttgrad(base_tt, grad_tt)

    print(grad_mats[0]) # First global grad matrix in batch
    # Grad entries are 105 = 35 * 3 (product of all TT ranks and 
    #                                number of cores defining TT matrix)