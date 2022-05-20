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
    def __init__(self, x, y):
        super(DataGenerator, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return np.shape(self.x)[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @staticmethod
    def collate(samples):
        batch_x, batch_y = [], []
        for item in samples:
            x, y = item
            batch_x.append(x)
            batch_y.append(y + 1)   # from [-1, 0, 1] to [0, 1, 2]

        batch_x = torch.tensor(np.array(batch_x)).float()
        batch_y = torch.tensor(np.array(batch_y)).long()
        return batch_x, batch_y
