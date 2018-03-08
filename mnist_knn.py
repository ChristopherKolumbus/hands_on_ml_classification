import numpy as np
from sklearn.datasets import fetch_mldata


def main():
    mnist = fetch_mldata('MNIST original')
    X_train, X_test, y_train, y_test = split_mnist_sets(mnist)
    X_train, y_train = shuffle_training_data(X_train, y_train)


def split_mnist_sets(mnist):
    X, y = mnist['data'], mnist['target']
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    return X_train, X_test, y_train, y_test


def shuffle_training_data(X_train, y_train, seed=42):
    np.random.seed(seed)
    shuffle_index = np.random.permutation(len(X_train))
    return X_train[shuffle_index], y_train[shuffle_index]


if __name__ == '__main__':
    main()