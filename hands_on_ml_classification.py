import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


def main():
    mnist = fetch_mldata('MNIST original')
    X, y = mnist['data'], mnist['target']
    some_digit = X[36000]
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    X_train, y_train = shuffle_data(X_train, y_train)
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    custom_cross_validation(X_train, y_train_5, sgd_clf)


def custom_cross_validation(X, y, clf, n_splits=3, random_state=42):
    skfolds = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    for train_index, test_index in skfolds.split(X, y):
        clone_clf = clone(clf)
        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_test_fold = X[test_index]
        y_test_fold = y[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))


def shuffle_data(X, y):
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same number of rows')
    shuffle_index = np.random.permutation(len(X))
    return X[shuffle_index], y[shuffle_index]


def show_digit(digit):
    digit_image = digit.reshape(28, 28)
    plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
