import scipy.io as sio
import numpy as np

class DataLoader:
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
