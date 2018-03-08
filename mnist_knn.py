from sklearn.datasets import fetch_mldata


def main():
    mnist = fetch_mldata('MNIST original')
    X_train, X_test, y_train, y_test = split_mnist_sets(mnist)


def split_mnist_sets(mnist):
    X, y = mnist['data'], mnist['target']
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    main()