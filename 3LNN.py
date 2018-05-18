import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
              learning_rate=0.1,
              batch_size=50,
              input_dim=784,
              hidden_dim=50,
              activation='sigmoid',
              num_classes=10,
              epochs=50
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

    sgd = SGD(lr=learning_rate)
    model.compile(
                  loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )

    history = model.fit(
              x_train,
              y_train,
              epochs=epochs,
              verbose=2,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              shuffle=True
             )

    return history


def plot_results(history, out_file):
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    type(val_acc)

    plt.plot(range(len(train_acc)), train_acc, color='blue')
    plt.plot(range(len(train_acc)), val_acc, color='orange')

    # add grid lines
    plt.grid(linestyle="dashed")

    # change limits to improve visibility
    plt.xlim((0, len(train_acc)-1))
    # plt.ylim((0, 105))

    # add labels
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")

    # add legends
    legend = mpatches.Patch(color='blue', label='Training')
    legend2 = mpatches.Patch(color='orange', label='Validation')
    plt.legend(handles=[legend, legend2])

    # save plot to file
    plt.savefig(out_file)
    plt.gcf().clear()


def main(argv):
    if len(argv) > 1:
        filename = argv[1]
        imgs, labels = get_data(filename)
    else:
        print("no input file, using default")
        imgs, labels = get_data()

    labels = one_hot(labels, 10)
    x_train, y_train, x_test, y_test = naive_split(imgs, labels)

    # batch_sizes = [x_train.shape[0]]
    batch_sizes = [1, 10, 50, x_train.shape[0]]
    learning_rates = [0.5, 1, 10]
    hidden_dims = [25, 50, 100]

    # history = run_model(x_train, y_train, x_test, y_test, learning_rate=0.5, batch_size=4000, hidden_dim=50, epochs=50)
    # plot_results(history, 'test')

    done = 0
    total = len(batch_sizes)*len(learning_rates)*len(hidden_dims)
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for hidden_dim in hidden_dims:
                print(done, "of", total)
                history = run_model(
                                    x_train,
                                    y_train,
                                    x_test,
                                    y_test,
                                    learning_rate=learning_rate,
                                    batch_size=batch_size,
                                    hidden_dim=hidden_dim
                                    )
                plot_results(history, "result"+str(done))
                done += 1


if __name__ == '__main__':
    main(sys.argv)
