import tensorflow as tf
import numpy as np
import os

from data.loader import simulation_loader
from utils.net_tool import intent_info


def restore_and_evaluation(pb_file, data_x, data_y):
    g = tf.Graph()
    with g.as_default():
        graph_def = tf.GraphDef()
        with open(pb_file, 'rb') as f:
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    with tf.Session(graph=g) as sess:
        tf.global_variables_initializer().run()

        x = sess.graph.get_tensor_by_name('x:0')
        y = sess.graph.get_tensor_by_name('y:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        accuracy = sess.graph.get_tensor_by_name('accuracy:0')
        logits_op = sess.graph.get_tensor_by_name('logits:0')

        logits = sess.run(logits_op, feed_dict={x: data_x, y: data_y, keep_prob: 1.0})
        train_accuracy = accuracy.eval(feed_dict={x: data_x, y: data_y, keep_prob: 1.0})

        intent = intent_info(logits, data_y)
    return train_accuracy, intent


def get_current_burn_in(data_x, data_y, tn):
    # print('data_x.shape[1]:', data_x.shape[1], '\ntn:', tn)
    if tn > data_x.shape[1]:
        print('please keep burn-in')
        return None

    data_x[:, tn:] = 0
    return data_x, data_y


def evaluation(model_path, test_data_path, tn, t, sdt, mdt, scale):

    model_file_list = np.array(sorted(os.listdir(model_path)))
    test_data_list = np.array(sorted(os.listdir(test_data_path)))

    models_accuracy = []
    models_intent = []
    for model_name in model_file_list:
        model_accuracy = []
        model_intent = []
        pb_file = '{}/{}'.format(model_path, model_name)

        for test_name in test_data_list:
            test_file = '{}/{}'.format(test_data_path, test_name)
            test_x, test_y = simulation_loader.get_test_data(test_file, t=t, sdt=sdt, mdt=mdt, scale=scale)
            # print(test_x.shape)

            # test_x, test_y = get_current_burn_in(test_x, test_y, tn)
            test_accuracy, intent = restore_and_evaluation(pb_file=pb_file, data_x=test_x, data_y=test_y)

            model_accuracy.append(test_accuracy)
            model_intent.append(intent)

        models_accuracy.append(model_accuracy)
        models_intent.append(model_intent)

    return np.array(models_accuracy), np.array(models_intent)


if __name__ == '__main__':
    # t: the actual burn-in time

    model_path = '../outputs/models/gamma'
    test_data_path = '../data/test/gamma'
    t, sdt, mdt = 2000, 50, 250
    scale = 1
    s, e = 9, 10

    # model_path = '../outputs/models/wiener'
    # test_data_path = '../data/test/wiener'
    # t, sdt, mdt = 200, 2, 10
    # scale = 1e6
    # s, e = 21, 22

    for tn in np.arange(start=s, stop=e, step=1):
        models_accuracy, models_intent = evaluation(model_path, test_data_path, tn, t=t, sdt=sdt, mdt=mdt, scale=scale)

        models_mean_accuracy = np.mean(models_accuracy, axis=1)

        for model_i in range(models_intent.shape[0]):
            print('tests accuracy/model{}:\n {} => {}\n'.format(model_i, models_accuracy[model_i],
                                                                models_mean_accuracy[model_i]))
            print('intent_info_tests/model{}: weak normal fp, fn\n {}\n\n'.format(model_i, models_intent[model_i]))


