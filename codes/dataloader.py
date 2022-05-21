import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGDataset:
    def __init__(self):
        self.EEG_x = sio.loadmat('SEED-III/EEG_X.mat')['X'][0]  # 15*3394*310
        self.EEG_y = sio.loadmat('SEED-III/EEG_Y.mat')['Y'][0]  # 15*3394*1
        self.num_dataset = len(self.EEG_x)  # 15
        print('%d datasets loaded from SEED-III.' % self.num_dataset)

    def leave_one_dataset(self, leave_which: int):
        if leave_which not in range(self.num_dataset):
            print('Leave-one error: out of index!')
            return

        train_index = np.delete(np.arange(self.num_dataset), leave_which)
        test_index = leave_which
        x_train = np.concatenate(self.EEG_x[train_index])
        y_train = np.concatenate(self.EEG_y[train_index])
        x_test = self.EEG_x[test_index]
        y_test = self.EEG_y[test_index]

        return x_train, np.ravel(y_train), x_test, np.ravel(y_test)


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
