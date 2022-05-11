import sklearn
import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.io as sio


class DataLoader:
    def __init__(self):
        self.EEG_x = sio.loadmat('SEED-III/EEG_X.mat')['X'][0]  # 15*3394*310
        self.EEG_y = sio.loadmat('SEED-III/EEG_Y.mat')['Y'][0]  # 15*3394*1
        self.num_dataset = len(self.EEG_x)  # 15
        print('%d datasets loaded from SEED-III.' % self.num_dataset)

    def LDA_dr(self):
        index_all = np.arange(15)
        x_ = np.concatenate(self.EEG_x[index_all])
        y_ = np.concatenate(self.EEG_y[index_all]).ravel()
        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(x_, y_)
        self.EEG_x = lda.transform(x_).reshape((15, 3394, 2))
        print('LDA completed!')

    def PCA(self):
        index_all = np.arange(15)
        x_ = np.concatenate(self.EEG_x[index_all])
        y_ = np.concatenate(self.EEG_y[index_all]).ravel()

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

        return x_train, y_train, x_test, y_test


def cross_validation(data: DataLoader, model: svm.SVC):
    print('[Cross validation] CV start!')
    cv_scores = []
    for dataset_no in range(data.num_dataset):
        x_train, y_train, x_test, y_test = data.leave_one_dataset(dataset_no)
        model.fit(x_train, np.ravel(y_train))
        acc = model.score(x_test, np.ravel(y_test))
        print('[Cross validation] Test on dataset %d: Acc %.4f' % (dataset_no, acc))
        cv_scores.append(acc)
    print('[Cross validation] Completed! Mean acc: %.4f\n' % (sum(cv_scores) / len(cv_scores)))


if __name__ == '__main__':
    models = (
        # Various kernels can be used here.
        # Hyper-parameters need to be adjusted.
        svm.SVC(kernel='linear', C=1, decision_function_shape='ovo', max_iter=300000),  # 1
        svm.SVC(kernel="rbf", gamma='scale', C=1, decision_function_shape='ovo', max_iter=300000),  # 2
        svm.SVC(kernel="poly", degree=3, gamma="auto", C=1, decision_function_shape='ovo', max_iter=300000)     # 3
    )
    eeg_data = DataLoader()
    eeg_data.LDA_dr()
    for model in models:
        cross_validation(eeg_data, model)



