import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_STATES = 310
HIDDEN_SIZE = 512

LR = 1e-3

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
        self.linear1 = nn.Linear(NUM_STATES, HIDDEN_SIZE)
        self.linear2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.lstm = nn.LSTMCell(HIDDEN_SIZE, 256)

        self.clf = nn.Linear(256, 3)

        self.apply(weights_init)
        self.clf.weight.data = normalized_columns_initializer(self.clf.weight.data, 0.01)
        self.clf.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, inputs):
        s, (hx, cx) = inputs
        x = F.elu(self.linear1(s))
        x = F.elu(self.linear2(x))

        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        logits = self.clf(x)

        return logits, (hx, cx)
