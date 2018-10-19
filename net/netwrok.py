from utils import net_tool
import tensorflow as tf
from net import layer


def create_network_mnist(x, keep_prob=1.0):
    activation_fun_conv = tf.nn.relu
    activation_fun_fc = tf.nn.relu

    conv1 = activation_fun_conv(layer.conv2d_layer(x, [5, 5, 1, 32], [32]))
    pool1 = layer.maxpool_layer(conv1)
    conv2 = activation_fun_fc(layer.conv2d_layer(pool1, [5, 5, 32, 64], [64]))
    pool2 = layer.maxpool_layer(conv2)

    flat = tf.reshape(pool2, [-1, 7*7*64])
    fc_3 = activation_fun_fc(layer.fc_layer(flat, [7*7*64, 1024], [1024]))
    drop_3 = layer.dropout_layer(fc_3, keep_prob)

    fc_4 = layer.fc_layer(drop_3, [1024, 10], [10])
    logits = net_tool.softmax(fc_4)

    return logits


def create_network_fc_gamma(x, keep_prob=1.0):
    # 17 512
    activation_fun_fc = tf.nn.elu

    fc_1 = activation_fun_fc(layer.fc_layer(x, [9, 256], [256]))
    drop_1 = layer.dropout_layer(fc_1, keep_prob)

    fc_2 = activation_fun_fc(layer.fc_layer(drop_1, [256, 64], [64]))
    drop_2 = layer.dropout_layer(fc_2, keep_prob)

    fc = layer.fc_layer(drop_2, [64, 2], [2])
    logits = net_tool.softmax(fc)

    return logits


def create_network_fc_wiener(x, keep_prob=1.0):
    activation_fun_fc = tf.nn.elu

    fc_1 = activation_fun_fc(layer.fc_layer(x, [26, 512], [512]))
    drop_1 = layer.dropout_layer(fc_1, keep_prob)

    fc_2 = activation_fun_fc(layer.fc_layer(drop_1, [512, 128], [128]))
    drop_2 = layer.dropout_layer(fc_2, keep_prob)

    fc = layer.fc_layer(drop_2, [128, 2], [2])
    logits = net_tool.softmax(fc)

    return logits
