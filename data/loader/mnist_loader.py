# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np


def load_data():
    f = gzip.open('../data/mnist/mnist.pkl.gz', 'rb')
    # 50000 + 10000 + 10000 = (data, id)
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    return training_data, validation_data, test_data


def vectorized_result(label, class_num=10):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros(class_num)
    e[label] = 1.0
    return e


def get_train_data():
    tr_d, va_d, te_d = load_data()

    # train_x = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    train_x = np.array(tr_d[0])
    train_y = np.array([vectorized_result(label=y) for y in tr_d[1]])

    return train_x, train_y


def get_valid_data():
    tr_d, va_d, te_d = load_data()

    valid_x = np.array(va_d[0])
    valid_y = np.array([vectorized_result(label=y) for y in va_d[1]])

    return valid_x, valid_y


def get_test_data():
    tr_d, va_d, te_d = load_data()

    test_x = np.array(te_d[0])
    test_y = np.array([vectorized_result(label=y) for y in te_d[1]])

    return test_x, test_y


