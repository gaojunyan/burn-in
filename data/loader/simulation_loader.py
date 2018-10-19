# Standard library
import pickle
import gzip
import numpy as np
from utils.data_tool import create_validation_split, data_scale
from utils import file_tool
import simulation.simulation_visual as vis


def load_data(filepath):
    f = gzip.open(filepath, 'rb')
    # (2000, 17) = (data, id)
    data, labels = pickle.load(f, encoding='bytes')
    f.close()
    return data, labels


def vectorized_result(label, class_num=2):
    """Return a 2-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros(class_num)
    e[np.int(label)] = 1.0
    return e


def produce_measurement_data(data_x, data_y, t, sdt, mdt, save_file):
    """
    :param data_x: ndarray (1000, 81)
    :param t: the actual burn-in time
    :param sdt: time interval of simulation = 50
    :param mdt: time interval of measurement = 250/50
    :return: measurement data used to model training and testing
    """
    stn = data_x.shape[1]
    dtn = np.int(mdt / sdt)
    mtn = np.int(t / mdt) + 1

    # print('t, mdt, stn, dtn, mtn = ', t, mdt, stn, dtn, mtn)

    if (mtn-1)*dtn >= stn:
        print('some errors in the simulation data provided! ')
        return None

    measurement_data = []
    for i in range(data_x.shape[0]):
        unit_i = [data_x[i, j*dtn] for j in range(mtn)]
        measurement_data.append(unit_i)
    measurement_data = np.array(measurement_data)

    # print(data_y.shape)
    # weak_n = np.sum(data_y).astype(np.int)
    # normal_n = data_y.shape[0] - weak_n
    # print(weak_n, normal_n)

    # axis_x = vis.generate_axis_x(mtn-1, mdt)
    # vis.data_visualize(axis_x, measurement_data, weak_n, normal_n, save_as_img=save_file)

    # produce the interfere
    # measurement_data[:, -1] = data_x[:, mtn*dtn + 1]
    return measurement_data


def get_train_and_valid_data(filepath, t, sdt, mdt, scale):
    path = '../data/measurement_img'
    # file_tool.mkdir(path)

    data, labels = load_data(filepath)
    # print(data.shape, labels.shape, filepath)
    # print(data[0, :])

    data = produce_measurement_data(data_x=data, data_y=labels, t=t, sdt=sdt, mdt=mdt,
                                    save_file='{}/train.jpg'.format(path))

    train_idx, valid_idx = create_validation_split(num_data=len(labels), num_validation=0.1)

    valid_x = np.array(data[valid_idx])
    valid_y = np.array([vectorized_result(y) for y in labels[valid_idx]])

    train_x = np.array(data[train_idx])
    train_y = np.array([vectorized_result(y) for y in labels[train_idx]])

    if scale != 1:
        train_x = data_scale(train_x, scale)
        valid_x = data_scale(valid_x, scale)

    return train_x, train_y, valid_x, valid_y


def get_test_data(filepath, t, sdt, mdt, scale=1):
    path = '../data/measurement_data_img'
    # file_tool.mkdir(path)

    data, labels = load_data(filepath)
    # print(data.shape, labels.shape, filepath)

    data = produce_measurement_data(data_x=data, data_y=labels, t=t, sdt=sdt, mdt=mdt,
                                    save_file='{}/test.jpg'.format(path))

    test_x = np.array(data)
    test_y = np.array([vectorized_result(y) for y in labels])

    if scale != 1:
        test_x = data_scale(test_x, scale)

    return test_x, test_y

