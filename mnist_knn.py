import os

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.ndimage import shift
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
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
    some_digit = X_train[40000]
    show_digit(some_digit)
    some_digit_shifted = shift_digit(some_digit, (10, 0))
    show_digit(some_digit_shifted)


def shift_digit(digit, amount):
    digit_image = reshape_digit(digit, mode='to_img')
    digit_image_shifted = shift(digit_image, amount)
    return reshape_digit(digit_image_shifted, mode='to_vector')


def show_digit(digit):
    digit_image = reshape_digit(digit, mode='to_img')
    plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()


def reshape_digit(digit, mode):
    if mode == 'to_img':
        return digit.reshape(28, 28)
    elif mode == 'to_vector':
        return digit.reshape(digit.size)


def grid_search_model(X_train, y_train):
    knn_clf = KNeighborsClassifier()
    parameters = {'n_neighbors': [5, 6, 7, 8], 'weights': ['distance']}
    grid_search = GridSearchCV(knn_clf, parameters, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train.astype(np.float64), y_train)
    best_estimator = grid_search.best_estimator_
    cv_results = grid_search.cv_results_
    for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
        print(mean_score, params)
    model_handler = ModelHandler(r'.\models')
    model_handler.save(best_estimator, 'knn_clf')



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