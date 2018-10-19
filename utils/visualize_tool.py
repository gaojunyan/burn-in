import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils import file_tool


def save_as_pkl(obj, save_file):
    with open(save_file, 'wb') as f:
        pickle.dump(obj, f)


def read_from_pkl(save_file):
    with open(save_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def visualize(y, title, save_file, save_name, color='k'):
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    x = np.arange(len(y))

    plt.plot(x, y, color=color, linewidth=1.0)
    plt.savefig('{}/{}'.format(save_file, save_name))
    plt.show()
    # plt.close()


def save_as_img(pklfile, title, imgfile, imgname, ):
    file_tool.mkdir(imgfile)
    data = read_from_pkl(pklfile)

    visualize(y=data, title=title, save_file=imgfile, save_name=imgname)