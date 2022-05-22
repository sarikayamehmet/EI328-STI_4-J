import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
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
        self.linear2 = nn.Linear(256, 64)
        self.clf = nn.Linear(64, 3)

        self.apply(weights_init)

    def forward(self, inputs):  # inputs: (batch_size, seq_len, FEATURE_DIM)
        output, (hx, cx) = self.lstm(inputs)    # hx: (2, batch_size, HIDDEN_SIZE)
        hx_L = hx[-2]
        hx_R = hx[-1]
        x = torch.cat((hx_L, hx_R), dim=1)  # x: (batch_size, HIDDEN_SIZE * 2)
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        logits = self.clf(x)

        return logits

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN_net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor
        self.lstm = nn.LSTM(FEATURE_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        # Label classifier
        self.label_clf = nn.Sequential()
        self.label_clf.add_module('lc_fc1', nn.Linear(HIDDEN_SIZE * 2, 256))
        self.label_clf.add_module('lc_bn1', nn.BatchNorm1d(256))
        self.label_clf.add_module('lc_elu1', nn.ELU(inplace=True)) 
        self.label_clf.add_module('lc_drop1', nn.Dropout(0.5))
        self.label_clf.add_module('lc_fc2', nn.Linear(256, 64))
        self.label_clf.add_module('lc_bn2', nn.BatchNorm1d(64))
        self.label_clf.add_module('lc_elu2', nn.ELU(inplace=True)) 
        self.label_clf.add_module('lc_fc3', nn.Linear(64, 3))
        # Domain classifier
        self.domain_clf = nn.Sequential()
        self.domain_clf.add_module('dc_fc1', nn.Linear(HIDDEN_SIZE * 2, 256))
        self.domain_clf.add_module('dc_bn1', nn.BatchNorm1d(256))
        self.domain_clf.add_module('dc_elu1', nn.ELU(inplace=True))
        self.domain_clf.add_module('dc_fc2', nn.Linear(256, 1))

    def forward(self, inputs, alpha=1):
        # Feature extractor
        output, (hx, cx) = self.lstm(inputs)    # hx: (2, batch_size, HIDDEN_SIZE)
        hx_L = hx[-2]
        hx_R = hx[-1]
        feature = torch.cat((hx_L, hx_R), dim=1)    # feature: (batch_size, HIDDEN_SIZE * 2)
        # Label classifier
        label_logits = self.label_clf(feature)
        # Domain classifier
        reversed_feature = ReverseLayerF.apply(feature, alpha)
        domain_logits = self.domain_clf(reversed_feature)

        return label_logits, domain_logits
