import argparse
import os
import random
from os.path import join

import numpy as np
import scipy
import sklearn.metrics
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader

from dataloader import DataGenerator, EEGDataset
from models import LSTM_net


def parse_args():
    parser = argparse.ArgumentParser()

    p = parser.add_argument_group("General")
    p.add_argument("--device", type=str, default='cpu')

    p = parser.add_argument_group("Model")
    p.add_argument("--load_path", type=str, default='./saved_models/TCA')
    p.add_argument("--save_path", type=str, default='./saved_models/TCA')

    p = parser.add_argument_group("Train")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p = parser.add_argument_group("Predict")
    p.add_argument("--predict_only", default=False, action='store_true')

    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class TCA:
    def __init__(self, kernel_type='primal', dim=310, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def kernel(self, ker, X1, X2, gamma):
        K = None
        if not ker or ker == 'primal':
            K = X1
        elif ker == 'linear':
            if X2 is not None:
                K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
            else:
                K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
        elif ker == 'rbf':
            if X2 is not None:
                K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
            else:
                K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
        return K

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = self.kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    seed_everything(args.seed)

    eeg_data = EEGDataset()
    tca = TCA(kernel_type='primal', dim=310)
    models = [LSTM_net().to(device) for i in range(15)]
    loss_func = nn.CrossEntropyLoss()
    total_y_true, total_y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
    for idx, model in enumerate(models):
        x_train, y_train, x_test, y_test = eeg_data.leave_one_dataset(idx)
        x_train, x_test = tca.fit(x_train, x_test)
        train_data = DataGenerator(x_train, y_train, seq_len=8)
        test_data = DataGenerator(x_test, y_test, seq_len=8)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, args.batch_size)

        if not args.predict_only:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            total_steps = len(train_loader)
            print('[TCA][Leave %d] Train begin!' % idx)
            model.train()
            for ep in range(1, args.epoch + 1):
                for i, batch in enumerate(train_loader):
                    batch_x, batch_y = batch
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    logits = model(batch_x)
                    loss = loss_func(logits, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print(f'\r[TCA][Leave {idx}][Epoch {ep}/{args.epoch}] > {i + 1}/{total_steps} Loss: {loss.item():.3f}', end='')
            
            path = join(args.save_path, 'model_leave%d.bin' % idx)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model.state_dict(), path)
            print()
        
        print('[TCA][Leave %d] Test begin!' % idx)
        path = join(args.save_path, 'model_leave%d.bin' % idx)
        model.load_state_dict(torch.load(path))
        model.eval()
        y_true, y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
        for i, batch in enumerate(test_loader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.data.numpy() - 1

            logits = model(batch_x)
            logits = logits.data.cpu().numpy()
            pred = np.argmax(logits, axis=1) - 1
            y_true = np.append(y_true, batch_y)
            y_pred = np.append(y_pred, pred)

        acc = accuracy_score(y_true, y_pred)
        print('[TCA][Leave %d] Test done! On dataset #%d: Acc %.4f' % (idx, idx, acc))
        total_y_true = np.append(total_y_true, y_true)
        total_y_pred = np.append(total_y_pred, y_pred)

    acc = accuracy_score(total_y_true, total_y_pred)
    print('[TCA] All tests done! Total acc: %.4f' % acc)
    disp = ConfusionMatrixDisplay.from_predictions(
        total_y_true, 
        total_y_pred, 
        normalize='all', 
        values_format='.3f'
    )
    plt.savefig('figures/TCA.png')
    plt.show()
