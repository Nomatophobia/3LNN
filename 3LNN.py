import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import KFold

import sys


def get_data(filename='data_tp1'):
    data = np.genfromtxt(filename, delimiter=',')
    labels = data[:, 0]
    imgs = data[:, 1:]
    return imgs, labels


def split_data(input_data, folds=5):
    print('Splitting data in', folds, 'folds')
    kf = KFold(n_splits=folds, shuffle=True)
    return(kf.split(input_data))


def naive_split(input_data, labels, train_size=4000):
    x_train = input_data[:train_size]
    y_train = labels[:train_size]

    x_test = input_data[train_size:]
    y_test = labels[train_size:]

    return (x_train, y_train, x_test, y_test)


def main(argv):
    if len(argv) > 1:
        filename = argv[1]
        imgs, labels = get_data(filename)
    else:
        print("no input file, using default")
        imgs, labels = get_data()

    x_train, y_train, x_test, y_test = naive_split(imgs, labels)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # split_data(input_data)


if __name__ == '__main__':
    main(sys.argv)
