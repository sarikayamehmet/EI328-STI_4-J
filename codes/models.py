import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

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


class grl_func(Function):
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
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64), 
            nn.ELU(inplace=True), 
            nn.Linear(64, 3)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 256),
            nn.BatchNorm1d(256), 
            nn.ELU(inplace=True),
            nn.Linear(256, 1)   # source 0, target 1
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


class ADDA_extractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(FEATURE_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True)

    def forward(self, inputs):
        output, (hx, cx) = self.lstm(inputs)  # hx: (2, batch_size, HIDDEN_SIZE)
        hx_L = hx[-2]
        hx_R = hx[-1]
        features = torch.cat((hx_L, hx_R), dim=1)   # features: (batch_size, HIDDEN_SIZE * 2)
        return features

class ADDA_classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 256),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64), 
            nn.ELU(inplace=True), 
            nn.Linear(64, 3)
        )

    def forward(self, inputs):
        logits = self.classifier(inputs)
        return logits

class ADDA_discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 256),
            nn.BatchNorm1d(256), 
            nn.ELU(inplace=True),
            nn.Linear(256, 1)   # source 0, target 1
        )
    
    def forward(self, inputs):
        logits = self.discriminator(inputs)
        return logits

class ADDAmodel:
    def __init__(self, lambda_=0.) -> None:
        self.src_extractor = ADDA_extractor()
        self.tar_extractor = ADDA_extractor()
        self.label_classifier = ADDA_classifier()
        self.domain_discriminator = ADDA_discriminator()
        self.grl = GRL(lambda_=lambda_)

    def update_para(self):
        'Copy parameters from src_extractor to tar_extractor.'
        self.tar_extractor.load_state_dict(self.src_extractor.state_dict())

    def set_lambda(self, lambda_):
        self.grl.set_lambda(lambda_)

    def save_model(self, path: str):
        torch.save({
            'src_extractor': self.src_extractor.state_dict(), 
            'tar_extractor': self.tar_extractor.state_dict(), 
            'label_classifier': self.label_classifier.state_dict(), 
            'domain_discriminator': self.domain_discriminator.state_dict()
            },
            path
        )

    def load_model(self, path: str):
        state_dict = torch.load(path)
        self.src_extractor.load_state_dict(state_dict['src_extractor'])
        self.tar_extractor.load_state_dict(state_dict['tar_extractor'])
        self.label_classifier.load_state_dict(state_dict['label_classifier'])
        self.domain_discriminator.load_state_dict(state_dict['domain_discriminator'])

    def to_device(self, device):
        self.src_extractor.to(device)
        self.tar_extractor.to(device)
        self.label_classifier.to(device)
        self.domain_discriminator.to(device)
        self.grl.to(device)
