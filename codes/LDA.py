import numpy as np
from sklearn import svm
from dataloader import DataLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def cross_validation(data: DataLoader, model: svm.SVC):
    print('[Cross validation] CV start!')
    cv_scores = []
    for dataset_no in range(data.num_dataset):
        x_train, y_train, x_test, y_test = data.leave_one_dataset(dataset_no)
        # train SVM
        lda = LinearDiscriminantAnalysis()
        x_train = lda.fit_transform(x_train, y_train)
        model.fit(x_train, y_train)
        # test on SVM
        x_test = lda.transform(x_test)
        acc = model.score(x_test, y_test)
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
    for model in models:
        cross_validation(eeg_data, model)
