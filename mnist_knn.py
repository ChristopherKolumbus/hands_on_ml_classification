import os

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


class ModelHandler:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

    def save(self, model, filename):
        joblib.dump(model, os.path.join(self.models_dir, filename))

    def load(self, filename):
        return joblib.load(os.path.join(self.models_dir, filename))

    def delete(self, filename):
        os.remove(os.path.join(self.models_dir, filename))


def main():
    mnist = fetch_mldata('MNIST original')
    X_train, X_test, y_train, y_test = split_mnist_sets(mnist)
    X_train, y_train = shuffle_training_data(X_train, y_train)
    knn_clf = KNeighborsClassifier()
    #cv_score = cross_val_score(knn_clf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    #print(cv_score)


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