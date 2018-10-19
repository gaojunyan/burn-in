import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def fc(x, w):
    return tf.matmul(x, w)


def dropout(x, p):
    return tf.nn.dropout(x, keep_prob=p)


def softmax(x):
    return tf.nn.softmax(x)


# y with the shape '[batch_size, 10]'
def cross_entropy(logits, y):
    tmp_sum = -1 * tf.reduce_sum(y * tf.log(logits), reduction_indices=[1])
    loss = tf.reduce_mean(tmp_sum)

    return loss


def accuracy_(logits, y):
    correct_preds = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    return accuracy


def intent_info(logits, y):
    normal_p, weak_p, false_positive, false_negative = -1, -1, 0, 0

    preds = np.argmax(logits, axis=1)
    true_y = np.argmax(y, axis=1)

    # print(preds, '\n\n', true_y)

    for i in range(preds.shape[0]):
        if true_y[i] == preds[i]:
            continue
        if (true_y[i] == 0) and (true_y[i] != preds[i]):
            false_negative += 1
        else:
            false_positive += 1

    weak_p = np.sum(preds)
    normal_p = logits.shape[0] - weak_p

    return [weak_p, normal_p, false_positive, false_negative]
