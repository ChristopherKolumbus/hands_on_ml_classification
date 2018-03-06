import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_mldata


def main():
    mnist = fetch_mldata('MNIST original')
    X, y = mnist['data'], mnist['target']
    show_digit(X, y, 36392)


def show_digit(X, y, index):
    digit = X[index]
    label = y[index]
    print(label)
    digit_image = digit.reshape(28, 28)
    plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
