import numpy as np
import pickle


# generate the observation time for all the unites
def generate_gamma_path(unit_num, time_num, g, v, dt):
    paths = []
    shape = g * dt
    scale = v
    for unit_i in range(unit_num):
        pathi = []
        pathi.append(0)
        for time_j in range(time_num):
            degradation = pathi[time_j] + np.random.gamma(shape, scale)
            pathi.append(degradation)
        paths.append(pathi)
    return np.array(paths)


def generate_gaussian_path(unit_num, time_num, loc, scale, theta, sigma, dt):
    """
    :param unit_num: int, the number of paths to be produced
    :param time_num: int, the length of each path {t0, t1, t2, ..., tn}
    :param loc: float, used in Normal distribution 'N(loc, scale)'
    :param scale: float, used in Normal distribution 'N(loc, scale)'
    :param theta: float, used in the gamma process 'W(t) = theta*t + sigma*B(t)' and 'B(t)~ N(loc, scale)'
    :param sigma: float, used in the gamma process 'W(t) = theta*t + sigma*B(t)' and 'B(t)~ N(loc, scale)'
    :param dt: float, time interval of two measurements
    :return: ndarray, paths produced
    """
    paths = []
    for unit_i in range(unit_num):
        pathi = []
        pathi.append(0)
        brown_move_t = 0
        for time_j in range(time_num):
            brown_move_t += np.random.normal(loc, scale)
            degradation = theta * dt * (time_j + 1) + brown_move_t * sigma
            pathi.append(degradation)
        paths.append(pathi)

    return np.array(paths)


# wrt. gamma process
def get_gamma_data(weak_n, normal_n, t_p, dt):
    # weak_n, normal_n, t_p, dt = 5, 95, 16, 250
    g1, g2, v1, v2 = 0.05, 0.04, 0.06, 0.05
    D_weak = generate_gamma_path(weak_n, t_p, g1, v1, dt)
    D_normal = generate_gamma_path(normal_n, t_p, g2, v2, dt)

    return D_weak, D_normal


# wrt.wiener process
def get_wiener_data(weak_n, normal_n, t_p, dt):
    # weak_n, normal_n, t_p, dt = 5, 95, 12, 100
    loc, scale = 0, 1
    theta1, theta2 = 8.4832*1e-9, 2.6875*1e-8
    sigma1, sigma2 = 6.002*1e-8, 6.002*1e-8
    # theta1, theta2 = 3.4832*1e-8, 8.6875*1e-8
    # sigma1, sigma2 = 9.002*1e-7, 9.002*1e-7

    D_weak = generate_gaussian_path(weak_n, t_p, loc, scale, theta2, sigma2, dt)
    D_normal = generate_gaussian_path(normal_n, t_p, loc, scale, theta1, sigma1, dt)
    return D_weak, D_normal


# save data with class label as .pkl file
def save_data_pkl(save_file, weak_n, normal_n, D_weak, D_normal):
    label_weak = np.ones(weak_n)
    label_norm = np.zeros(normal_n)
    labels = np.concatenate([label_weak, label_norm], axis=0)
    data = np.concatenate([D_weak, D_normal], axis=0)
    with open(save_file, 'wb') as f:
            pickle.dump((data, labels), f)


def generate_simulation_data(weak_train_n, normal_train_n, weak_test_n, normal_test_n,
                             t_n, dt, save_file, process):
    # weak_n, normal_n, t_n, dt = 5, 95, 16, 250 / 1000, 1000, 12, 100

    get_data_func = get_gamma_data if process == 'gamma' else get_wiener_data

    weak_train, normal_train = get_data_func(weak_train_n, normal_train_n, t_n, dt)
    save_data_pkl('{}/{}_data_train.pkl'.format(save_file, process),
                  weak_train_n, normal_train_n, weak_train, normal_train)

    weak_test, normal_test = get_data_func(weak_test_n, normal_test_n, t_n, dt)
    save_data_pkl('{}/{}_data_test.pkl'.format(save_file, process),
                  weak_test_n, normal_test_n, weak_test, normal_test)

