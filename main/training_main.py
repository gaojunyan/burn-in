import tensorflow as tf
from utils import net_tool, data_tool, file_tool, visualize_tool
from data.loader import simulation_loader
from net import netwrok
import numpy as np

# graph save as pb file
from tensorflow.python.framework import graph_util


def factory(train_data_file, test_data_file, path_results, path_models, net_type, t, sdt, mdt):

    # build graph
    g = tf.Graph()
    create_network_fc = netwrok.create_network_fc_gamma if net_type == 'gamma' else netwrok.create_network_fc_wiener
    scale = 1 if net_type == 'gamma' else 1e6

    with g.as_default():
        x = tf.placeholder(tf.float32, [None, 9], name='x')
        y = tf.placeholder(tf.float32, [None, 2], name='y')
        keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        logits = tf.multiply(create_network_fc(x, keep_prob), 1, name='logits')
        loss = net_tool.cross_entropy(logits, y)
        train_step = tf.train.AdagradOptimizer(1e-4).minimize(loss)

        accuracy = tf.add(net_tool.accuracy_(logits, y), 0, name='accuracy')

    # run graph
    with tf.Session(graph=g) as sess:
        tf.global_variables_initializer().run()

        epochs = 50000
        batch_size = 128

        train_x, train_y, valid_x, valid_y = simulation_loader.get_train_and_valid_data(train_data_file, t=t,
                                                                                        sdt=sdt, mdt=mdt, scale=scale)
        test_x, test_y = simulation_loader.get_test_data(test_data_file, t=t,
                                                         sdt=sdt, mdt=mdt, scale=scale)

        # print(train_x[0, :])
        print('load data success, train_x:{}, train_y:{}, test{}'.format(train_x.shape, train_y.shape, test_x.shape))

        train_data_len = len(train_x)
        training_loss = []
        training_accuracy = []
        batch_idx_dis = []

        mean_loss = 0
        graph_def = tf.get_default_graph().as_graph_def()
        for i in range(epochs):
            total_step = 0
            total_loss = 0
            for batch_idx in data_tool.generate_batch_indices_shuffle(train_data_len, batch_size):
                sess.run(train_step, feed_dict={
                            x: train_x[batch_idx], y: train_y[batch_idx], keep_prob: 0.5
                        })
                total_step += 1
                total_loss += sess.run(loss, feed_dict={
                            x: train_x[batch_idx], y: train_y[batch_idx], keep_prob: 1
                        })
                batch_idx_dis.append(np.sum((batch_idx < 500).astype(np.float32), axis=0))

            constant_graph = graph_util.convert_variables_to_constants(sess, graph_def, ['logits', 'accuracy'])
            with tf.gfile.GFile('{}/model_{}.pb'.format(path_models, str(i).zfill(5)), 'wb') as f:
                f.write(constant_graph.SerializeToString())

            train_accuracy = accuracy.eval(feed_dict={x: valid_x, y: valid_y, keep_prob: 1.0})
            training_accuracy.append(train_accuracy)
            mean_loss = (mean_loss + total_loss / total_step)/2
            training_loss.append(mean_loss)
            print('step %d, training accuracy %g, loss %f' % (i, train_accuracy, mean_loss))

        test_accuracy = accuracy.eval(feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
        print('test accuracy %g' % test_accuracy)

        visualize_tool.save_as_pkl(np.array(training_loss), '{}/training_loss.pkl'.format(path_results))
        visualize_tool.save_as_pkl(np.array(training_accuracy), '{}/training_accuracy.pkl'.format(path_results))
        visualize_tool.save_as_pkl(np.array(batch_idx_dis), '{}/batch_idx.pkl'.format(path_results))


def save_as_img(data_path, img_path):
    # visualize_tool.save_as_img(pklfile='{}/batch_idx.pkl'.format(data_path),
    #                            title='batch/training', imgfile=img_path, imgname='training_batch.jpg', )

    visualize_tool.save_as_img(pklfile='{}/training_accuracy.pkl'.format(data_path),
                               title='accuracy/training', imgfile=img_path, imgname='training_accuracy.jpg')

    visualize_tool.save_as_img(pklfile='{}/training_loss.pkl'.format(data_path),
                               title='loss/training', imgfile=img_path, imgname='training_loss.jpg')


if __name__ == '__main__':
    data_type = 'gamma'
    t, sdt, mdt = 2000, 50, 250

    # data_type = 'wiener'
    # t, sdt, mdt = 150, 3, 6

    # data file
    train_data_file = '../data/{}/{}_data_train.pkl.gz'.format(data_type, data_type)
    test_data_file = '../data/{}/{}_data_test.pkl.gz'.format(data_type, data_type)

    # save training results
    path_results = '../outputs/results/{}'.format(data_type)
    file_tool.mkdir(path_results)

    # save training models
    path_models = '../outputs/models/{}'.format(data_type)
    file_tool.mkdir(path_models)

    factory(train_data_file=train_data_file, test_data_file=test_data_file,
            path_results=path_results, path_models=path_models, net_type=data_type,
            t=t, sdt=sdt, mdt=mdt)

    img_path = '../outputs/img/{}'.format(data_type)
    file_tool.mkdir(img_path)

    save_as_img(data_path=path_results, img_path=img_path)