from data.loader import simulation_loader
import numpy as np
import os


def test_accuracy(data_x, data_y, t, cut_off_level, mdt):
    # weak:1   normal:0
    tn = np.int(t / mdt)
    degradations = np.array(data_x[:, tn])

    preds = (degradations > cut_off_level).astype(np.float)
    true_y = np.argmax(data_y, axis=1)

    false_positive, false_negative = 0, 0

    for i in range(preds.shape[0]):
        if true_y[i] == preds[i]:
            continue
        if (true_y[i] == 0) and (preds[i] != true_y[i]):
            false_negative += 1
        else:
            false_positive += 1

    mean_accuracy = np.mean(np.equal(preds, true_y).astype(np.float))
    return mean_accuracy, [false_positive, false_negative]


def evaluation(test_data_path, t, cut_off_level, sdt, mdt):
    test_data_list = np.array(sorted(os.listdir(test_data_path)))

    tests_accuracy = []
    intents_info = []
    for test_name in test_data_list:
        test_file = '{}/{}'.format(test_data_path, test_name)
        test_x, test_y = simulation_loader.get_test_data(test_file, t=t, sdt=sdt, mdt=mdt)

        # print(test_x.shape, test_y.shape)
        accuracy, intent = test_accuracy(test_x, test_y, t, cut_off_level, mdt)
        tests_accuracy.append(accuracy)
        intents_info.append(intent)

    return np.array(tests_accuracy), np.array(intents_info)


if __name__ == '__main__':
    test_data_path = '../data/test/gamma'
    t, cut_off_level, sdt, mdt = 2000, 4.7781, 50, 250

    tests_accuracy, intents_info = evaluation(test_data_path, t, cut_off_level, sdt, mdt)
    mean_accuracy = np.mean(tests_accuracy)

    print('tests accuracy => mean accuracy:\n{} =>{}\n\n'.format(tests_accuracy, mean_accuracy))
    print('intent info/test: fp, fn:\n{}'.format(intents_info))