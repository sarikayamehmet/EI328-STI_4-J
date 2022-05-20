from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay
from dataloader import EEGDataset

def cross_validation(data: EEGDataset, model: svm.SVC):
    print('[Cross validation] CV start!')
    cv_scores = []
    y_true, y_pred = [], []
    for dataset_no in range(data.num_dataset):
        x_train, y_train, x_test, y_test = data.leave_one_dataset(dataset_no)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        acc = model.score(x_test, y_test)
        print('[Cross validation] Test on dataset %d: Acc %.4f' % (dataset_no, acc))
        cv_scores.append(acc)
        y_true.append(y_test)
        y_pred.append(pred)
    print('[Cross validation] Completed! Mean acc: %.4f\n' % (sum(cv_scores) / len(cv_scores)))
    return np.array(y_true), np.array(y_pred)

if __name__ == '__main__':
    eeg_data = EEGDataset()
    models = (
        # Various kernels can be used here.
        # Hyper-parameters need to be adjusted.
        svm.SVC(kernel='linear', C=1, decision_function_shape='ovo', max_iter=300000),  # 1
        # svm.SVC(kernel="rbf", gamma='scale', C=1, decision_function_shape='ovo', max_iter=300000),  # 2
        # svm.SVC(kernel="poly", degree=3, gamma="auto", C=1, decision_function_shape='ovo', max_iter=300000)     # 3
    )
    for model in models:
        y_true, y_pred = cross_validation(eeg_data, model)
        y_true, y_pred = np.hstack(y_true), np.hstack(y_pred)
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, 
            y_pred, 
            normalize='all', 
            values_format='.3f'
        )
        plt.show()
