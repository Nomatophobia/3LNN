import numpy as np
import keras
import sys


def getData(filename='data_tp1'):
    return np.genfromtxt(filename, delimiter=',')


def main(argv):
    if len(argv) > 1:
        filename = argv[1]
        data = getData(filename)
    else:
        print("no input file, using default")
        data = getData()



if __name__ == '__main__':
    main(sys.argv)
