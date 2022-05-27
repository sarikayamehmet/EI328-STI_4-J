import argparse
import os
from os.path import join
import random
from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import torch
from models import LSTM_net
from dataloader import EEGDataset, DataGenerator
import torch.nn as nn
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()

    p = parser.add_argument_group("General")
    p.add_argument("--device", type=str, default='cpu')

    p = parser.add_argument_group("Model")
    p.add_argument("--load_path", type=str, default='./saved_models/CORAL')
    p.add_argument("--save_path", type=str, default='./saved_models/CORAL')

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


class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.real(np.dot(Xs, A_coral))
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit(Xs, Xt)
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    seed_everything(args.seed)

    eeg_data = EEGDataset()

    coral = CORAL()
    models = [LSTM_net().to(device) for i in range(15)]
    loss_func = nn.CrossEntropyLoss()
    total_y_true, total_y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
    total_y_truth, total_y_predict = np.empty(0, dtype=int), np.empty(0, dtype=int)
    for idx, model in enumerate(models):
        x_train, y_train, x_test, y_test = eeg_data.leave_one_dataset(idx)
        x_train = coral.fit(x_train, x_test)
        train_data = DataGenerator(x_train, y_train, seq_len=8)
        test_data = DataGenerator(x_test, y_test, seq_len=8)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, args.batch_size)

        if not args.predict_only:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            total_steps = len(train_loader)
            print('[CORAL][Leave %d] Train begin!' % idx)
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
                    print(
                        f'\r[CORAL][Leave {idx}][Epoch {ep}/{args.epoch}] > {i + 1}/{total_steps} Loss: {loss.item():.3f}',
                        end='')

            path = join(args.save_path, 'model_leave%d.bin' % idx)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model.state_dict(), path)
            print()

        print('[CORAL][Leave %d] Test begin!' % idx)
        # CORAL + KNN-1
        acc, y_predict = coral.fit_predict(x_train, y_train, x_test, y_test)
        print('[KNN-1][Leave %d] Test done! On dataset #%d: Acc %.4f' % (idx, idx, acc))
        total_y_true = np.append(total_y_truth, y_test)
        total_y_pred = np.append(total_y_predict, y_predict)

        # # CORAL + LSTM
        # path = join(args.save_path, 'model_leave%d.bin' % idx)
        # model.load_state_dict(torch.load(path))
        # model.eval()
        # y_true, y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
        # for i, batch in enumerate(test_loader):
        #     batch_x, batch_y = batch
        #     batch_x = batch_x.to(device)
        #     batch_y = batch_y.data.numpy() - 1

        #     logits = model(batch_x)
        #     logits = logits.data.cpu().numpy()
        #     pred = np.argmax(logits, axis=1) - 1
        #     y_true = np.append(y_true, batch_y)
        #     y_pred = np.append(y_pred, pred)

        # acc = accuracy_score(y_true, y_pred)
        # print('[CORAL][Leave %d] Test done! On dataset #%d: Acc %.4f' % (idx, idx, acc))
        # total_y_true = np.append(total_y_true, y_true)
        # total_y_pred = np.append(total_y_pred, y_pred)

    acc = accuracy_score(total_y_truth, total_y_predict)
    print('[KNN-1] All tests done! Total acc: %.4f' % acc)

    # acc = accuracy_score(total_y_true, total_y_pred)
    # print('[CORAL] All tests done! Total acc: %.4f' % acc)
    disp = ConfusionMatrixDisplay.from_predictions(
        total_y_true,
        total_y_pred,
        normalize='all',
        values_format='.3f'
    )
    plt.savefig('figures/CORAL.png')
    plt.show()
