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


class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=0.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grl_func.apply(x, self.lambda_)


class DANN(nn.Module):
    def __init__(self, lambda_=0.):
        super(DANN, self).__init__()
        self.lstm = nn.LSTM(FEATURE_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True)

        self.task_classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 256),
            nn.ELU(),
            nn.Linear(256, 3)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 256),
            nn.ELU(),
            nn.Linear(256, 2)   # source 0, target 1
        )

        self.grl = GRL(lambda_=lambda_)

    def forward(self, inputs):
        output, (hx, cx) = self.lstm(inputs)  # hx: (2, batch_size, HIDDEN_SIZE)
        hx_L = hx[-2]
        hx_R = hx[-1]
        h = torch.cat((hx_L, hx_R), dim=1)  # h: (batch_size, HIDDEN_SIZE * 2)

        task_predict = self.task_classifier(h)

        x = self.grl(h)
        domain_predict = self.domain_classifier(x)

        return task_predict, domain_predict

    def set_lambda(self, lambda_):
        self.grl.set_lambda(lambda_)
