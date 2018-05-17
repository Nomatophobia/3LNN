import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.model_selection import KFold

import sys

DATA_DIR = 'data/data_tp1'


# Read data from file
def get_data(filename=DATA_DIR):
    data = np.genfromtxt(filename, delimiter=',')
    labels = data[:, 0]
    imgs = data[:, 1:]
    return imgs, labels


# Split data into folds
def split_data(input_data, folds=5):
    print('Splitting data in', folds, 'folds')
    kf = KFold(n_splits=folds, shuffle=True)
    return(kf.split(input_data))


# Split data naively into training and testing sets
def naive_split(input_data, labels, train_size=4000):
    x_train = input_data[:train_size]
    y_train = labels[:train_size]

    x_test = input_data[train_size:]
    y_test = labels[train_size:]

    return (x_train, y_train, x_test, y_test)


# Convert data to one hot encoding format
def one_hot(data, num_classes=None):
    return to_categorical(data, num_classes=num_classes)


# Define Neural Network model
def run_model(
                x_train,
                y_train,
                x_test,
                y_test,
                input_dim=784,
                hidden_dim=50,
                activation='sigmoid',
                num_classes=10
                ):

    model = Sequential()
    model.add(Dense(
                    hidden_dim,
                    activation=activation,
                    input_dim=input_dim,
                   ))

    model.add(Dense(
                    num_classes,
                    activation=activation,
                    input_dim=hidden_dim,
                   ))

    # sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.1)
    model.compile(
                  loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )

    score = model.fit(
              x_train,
              y_train,
              epochs=50,
              verbose=1,
              batch_size=128,
              validation_data=(x_test, y_test),
              shuffle=True
             )

    return score


def main(argv):
    if len(argv) > 1:
        filename = argv[1]
        imgs, labels = get_data(filename)
    else:
        print("no input file, using default")
        imgs, labels = get_data()

    labels = one_hot(labels, 10)
    x_train, y_train, x_test, y_test = naive_split(imgs, labels)
    history = run_model(x_train, y_train, x_test, y_test)
    print(history.history['acc'])


if __name__ == '__main__':
    main(sys.argv)
