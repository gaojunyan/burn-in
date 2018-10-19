import matplotlib.pyplot as plt
import utils.visualize_tool as vis
import numpy as np


def gamma_loss_vis():
    loss_file = '../outputs/results/gamma/training_loss.pkl'
    loss_data = vis.read_from_pkl(loss_file)

    num = np.int(len(loss_data) / 100)
    x = [i*100 for i in range(num)]
    y = [loss_data[i*100] for i in range(num)]

    axes = plt.gca()
    axes.set_xlabel('Epoch')
    axes.set_title('loss/training')
    plt.plot(x, y, color='k', linewidth=1.0)
    plt.savefig('../outputs/img/wiener_loss.jpg')
    plt.show()


def wiener_loss_vis():
    loss_file = '../outputs/results/wiener/training_loss.pkl'
    loss_data = vis.read_from_pkl(loss_file)
    y = loss_data[10:]
    axes = plt.gca()
    axes.set_xlabel('Epoch')

    axes.set_title('loss/training')
    plt.plot(y, color='k', linewidth=1.0)
    plt.savefig('../outputs/img/wiener_loss.jpg')
    plt.show()


def wiener_accuracy_vis():
    acc_file = '../outputs/results/wiener/training_accuracy.pkl'
    acc_data = vis.read_from_pkl(acc_file)
    axes = plt.gca()
    axes.set_xlabel('Epoch')

    axes.set_title('accuracy/training')
    plt.plot(acc_data, color='k', linewidth=1.0)
    plt.savefig('../outputs/img/wiener_accuracy.jpg')
    plt.show()


def gamma_accuracy_vis():
    acc_file = '../outputs/results/gamma/training_accuracy.pkl'
    acc_data = vis.read_from_pkl(acc_file)
    axes = plt.gca()
    axes.set_xlabel('Epoch')

    axes.set_title('accuracy/training')
    plt.plot(acc_data, color='k', linewidth=1.0)
    plt.savefig('../outputs/img/gamma_accuracy.jpg')
    plt.show()


def gamma_vis():
    axes = plt.gca()
    axes.set_xlabel('Epoch')

    loss_file = '../outputs/results/gamma/training_loss.pkl'
    loss_data = vis.read_from_pkl(loss_file)
    num = np.int(len(loss_data) / 100)
    x = [i*100 for i in range(num)]
    y = [loss_data[i*100] for i in range(num)]

    plt.plot(x, y, color='b', linewidth=1.0)

    acc_file = '../outputs/results/gamma/training_accuracy.pkl'
    acc_data = vis.read_from_pkl(acc_file)
    plt.plot(acc_data, color='r', linewidth=1.0)

    plt.savefig('../outputs/img/gamma.jpg')
    plt.show()


def wiener_vis():
    axes = plt.gca()
    axes.set_xlabel('Epoch')

    loss_file = '../outputs/results/wiener/training_loss.pkl'
    loss_data = vis.read_from_pkl(loss_file)
    num = np.int(len(loss_data) / 100)
    x = [i * 100 for i in range(num)]
    y = [loss_data[i * 100] for i in range(num)]

    plt.plot(x, y, color='b', linewidth=1.0)

    acc_file = '../outputs/results/wiener/training_accuracy.pkl'
    acc_data = vis.read_from_pkl(acc_file)
    plt.plot(acc_data, color='r', linewidth=1.0)

    plt.savefig('../outputs/img/wiener.jpg')
    plt.show()


if __name__ == '__main__':
    # gamma_loss_vis()
    # wiener_loss_vis()
    #
    # gamma_accuracy_vis()
    # wiener_accuracy_vis()
    gamma_vis()
    wiener_vis()

