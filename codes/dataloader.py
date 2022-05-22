import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset:
    def __init__(self):
        self.EEG_x = sio.loadmat('SEED-III/EEG_X.mat')['X'][0]  # (15, 3394, 310)
        self.EEG_y = sio.loadmat('SEED-III/EEG_Y.mat')['Y'][0]  # (15, 3394, 1)
        self.num_dataset = len(self.EEG_x)  # 15
        print('%d datasets loaded from SEED-III.' % self.num_dataset)

    def leave_one_dataset(self, leave_which: int):
        if leave_which not in range(self.num_dataset):
            print('Leave-one error: out of index!')
            return

        train_index = np.delete(np.arange(self.num_dataset), leave_which)
        test_index = leave_which
        x_train = np.concatenate(self.EEG_x[train_index])   # (15*3394, 310)
        y_train = np.concatenate(self.EEG_y[train_index])   # (15*3394, )
        x_test = self.EEG_x[test_index]     # (3394*310, 310)
        y_test = self.EEG_y[test_index]     # (3394, )

        return x_train, np.ravel(y_train), x_test, np.ravel(y_test)


class _EEGDataset:
    def __init__(self):
        self.EEG_x = sio.loadmat('SEED-III/EEG_X.mat')['X'][0]  # (15, ) 3394, 310
        self.EEG_y = sio.loadmat('SEED-III/EEG_Y.mat')['Y'][0]  # (15, ) 3394, 1
        self.EEG_d = np.zeros((15, 3394), dtype=int)
        # for i in range(15):
        #     self.EEG_d[i, :] = i
        self.num_dataset = len(self.EEG_x)  # 15
        print('%d datasets loaded from SEED-III.' % self.num_dataset)

    def leave_one_dataset(self, leave_which: int):
        if leave_which not in range(self.num_dataset):
            print('Leave-one error: out of index!')
            return
        self.EEG_d[leave_which, :] = 1

        train_index = np.delete(np.arange(self.num_dataset), leave_which)
        test_index = leave_which
        x_train = np.concatenate(self.EEG_x[train_index])   # (15*3394, 310)
        y_train = np.concatenate(self.EEG_y[train_index])   # (15*3394, )
        d_source = np.concatenate(self.EEG_d[train_index])
        x_test = self.EEG_x[test_index]     # (3394*310, 310)
        y_test = self.EEG_y[test_index]     # (3394, )
        d_target = self.EEG_d[test_index]

        return x_train, np.ravel(y_train), x_test, np.ravel(y_test), d_source, d_target


class DataGenerator(Dataset):
    def __init__(self, x, y, seq_len=8):
        super(DataGenerator, self).__init__()
        self.x = x
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return np.shape(self.x)[0]

    def __getitem__(self, idx):
        if idx + self.seq_len > self.__len__():
            seq_x = np.zeros((self.seq_len, np.shape(self.x)[1]))
            seq_x[:self.__len__() - idx] = self.x[idx:]
        else:
            seq_x = self.x[idx : idx + self.seq_len]
        seq_y = self.y[idx]

        seq_x = torch.tensor(seq_x).float()
        seq_y = torch.tensor(seq_y).long() + 1  # from [-1, 0, 1] to [0, 1, 2]
        return seq_x, seq_y


class DomainDataGenerator(Dataset):
    def __init__(self, x, y, ds, xt=None, dt=None, seq_len=8):
        super(DomainDataGenerator, self).__init__()
        self.x = x
        self.y = y
        self.ds = ds
        self.xt = xt
        self.dt = dt
        self.seq_len = seq_len

    def __len__(self):
        return np.shape(self.x)[0]

    def __getitem__(self, idx):
        if idx < self.seq_len:
            seq_x = np.zeros((self.seq_len, np.shape(self.x)[1]))
            seq_x[-idx-1:] = self.x[:idx+1]
        else:
            seq_x = self.x[idx - self.seq_len: idx]

        seq_y = self.y[idx]

        seq_x = torch.tensor(seq_x).float()
        seq_y = torch.tensor(seq_y).long() + 1  # from [-1, 0, 1] to [0, 1, 2]

        if self.xt is not None:
            x_target_len = self.xt.shape[0]     # 3394
            idx = idx % x_target_len
            if idx < self.seq_len:
                seq_xt = np.zeros((self.seq_len, self.xt.shape[1]))
                seq_xt[-idx-1:] = self.xt[:idx+1]
            else:
                seq_xt = self.xt[idx - self.seq_len: idx]
            seq_dt = self.dt[idx]

            seq_d = self.ds[idx]
            seq_d = torch.tensor(seq_d).long()
            seq_xt = torch.tensor(seq_xt).float()
            seq_dt = torch.tensor(seq_dt).long()
            return seq_x, seq_y, seq_d, seq_xt, seq_dt
        else:
            return seq_x, seq_y