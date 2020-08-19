import torch
import torch.nn as nn
from t3nsor.layers import TTLinear

class TTLinearSet(nn.Module):
    """
    Creates n TTLinear modules. Output of forward pass is the (column-wise) concatenation of the
    output of the n individual TTLinear modules.
    """
    def __init__(self, in_features=None, out_features=None, n_gates=4, bias=True, init=None, shape=None,
                 auto_shapes=True, d=3, tt_rank=8, auto_shape_mode='ascending',
                 auto_shape_criterion='entropy'):
        super(TTLinearSet, self).__init__()
        self.n_gates = n_gates
        self.in_features = in_features
        self.out_features = out_features
        self.gates = []
        for i in range(n_gates):
            cur_gate = TTLinear(in_features=in_features, out_features=out_features,
                                bias=bias, auto_shapes=auto_shapes, d=d, tt_rank=tt_rank,
                                init=init, shape=shape, auto_shape_mode=auto_shape_mode,
                                auto_shape_criterion=auto_shape_criterion)
            setattr(self, f'gate{i}', cur_gate)
            self.gates.append(cur_gate)

    def forward(self, x):
        batch_size, in_size = x.size()

        out = torch.zeros(batch_size, self.out_features * self.n_gates).to(x.device)
        for i in range(self.n_gates):
            cur_gate = self.gates[i]
            start_idx = i * self.out_features
            end_idx = (i + 1) * self.out_features
            out[:, start_idx:end_idx] = cur_gate(x)
        return out
