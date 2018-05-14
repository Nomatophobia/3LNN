import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import KFold

import sys


def getData(filename='data_tp1'):
    return np.genfromtxt(filename, delimiter=',')


def split_data(input_data, folds=5):
    print('Splitting data in', folds, 'folds')
    kf = KFold(n_splits=folds, shuffle=True)
    return(kf.split(input_data))


def main(argv):
    if len(argv) > 1:
        filename = argv[1]
        input_data = getData(filename)
    else:
        print("no input file, using default")
        input_data = getData()
    split_data(input_data)

if __name__ == '__main__':
    main(sys.argv)
