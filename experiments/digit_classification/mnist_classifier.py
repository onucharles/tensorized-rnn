import torch.nn.functional as F
from torch import nn
import sys

# sys.path.append("../")
# from t3nsor.layers import TTLinear
# from encoder.models.lstm import LSTM
# from encoder.models.tt_lstm import TTLSTM


class MNIST_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(MNIST_Classifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        out, (last_hidden, last_cell) = self.lstm(inputs)
        o = self.linear(out[:, -1, :])
        return F.log_softmax(o, dim=1)
