from torch import nn

class LRLinear(nn.Module):
    """
    Low-rank factorised linear module.
    """
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LRLinear, self).__init__()

        self.linear1 = nn.Linear(in_features=in_features,
                                    out_features=rank, bias=bias)
        self.linear2 = nn.Linear(in_features=rank,
                                 out_features=out_features, bias=bias)

    def forward(self, x):
        return self.linear2(self.linear1(x))