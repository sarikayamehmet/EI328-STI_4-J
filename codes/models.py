import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

FEATURE_DIM = 310
HIDDEN_SIZE = 256

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class LSTM_net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(FEATURE_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(HIDDEN_SIZE * 2, 256)
        self.clf = nn.Linear(256, 3)

        self.apply(weights_init)

    def forward(self, inputs):  # inputs: (batch_size, seq_len, FEATURE_DIM)
        output, (hx, cx) = self.lstm(inputs)    # hx: (2, batch_size, HIDDEN_SIZE)
        hx_L = hx[-2]
        hx_R = hx[-1]
        x = torch.cat((hx_L, hx_R), dim=1)  # x: (batch_size, HIDDEN_SIZE * 2)
        x = F.elu(self.linear1(x))
        logits = self.clf(x)

        return logits
