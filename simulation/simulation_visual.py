import numpy as np
import matplotlib.pyplot as plt
import pickle


def load_data(filepath):
    with open(filepath, 'rb') as f:
        data, labels = pickle.load(f)
    return data, labels


def generate_axis_x(t_p, dt):
    x = []
    for i in range(t_p+1):
        x.append(i * dt)
    return x


def data_visualize(x, data, weak_n, normal_n, save_as_img):
    axes = plt.gca()
    axes.set_xlabel('time')
    # axes.set_ylabel('degradation')
    for pathi in data[np.arange(weak_n)]:
        plt.plot(x, pathi, 'r')

    for pathi in data[np.arange(weak_n, weak_n+normal_n, step=1)]:
        plt.plot(x, pathi, 'g', linewidth=1)

    plt.savefig(save_as_img)
    plt.close()


def visualize(weak_n, normal_n, t_n, dt, filepath, save_as_img):
    # weak_n, normal_n, t_p, dt = 5, 95, 16, 250
    # filepath = 'simulation/gamma_data_test.pkl'

    data, labels = load_data(filepath)
    x = generate_axis_x(t_n, dt)
    data_visualize(x, data, weak_n, normal_n, save_as_img)


