import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import KFold

import sys

DATA_DIR = 'data/data_tp1'


def get_data(filename=DATA_DIR):
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


def define_model(
                x_train,
                y_train,
                x_test=None,
                y_test=None,
                input_shape=784,
                hidden_dim=50,
                activation='sigmoid',
                num_classes=10
                ):

    model = Sequential
    model.add(Dense(
                    hidden_dim,
                    activation=activation,
                    input_shape=input_shape,
                   ))

    model.add(Dense(
                    num_classes,
                    activation=activation,
                    input_shape=hidden_dim,
                   ))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
                  loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )

    model.fit(
              x_train,
              y_train,
              epochs=20,
              batch_size=128
              )

    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)


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
