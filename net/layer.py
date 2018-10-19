from utils import net_tool


def conv2d_layer(x, w_shape, b_shape=None):
    w = net_tool.weight_variable(w_shape)
    output = net_tool.conv_2d(x, w)

    if not b_shape:
        b = net_tool.bias_variable(b_shape)
        output = output + b

    return output


def fc_layer(x, w_shape, b_shape=None):
    w = net_tool.weight_variable(w_shape)
    output = net_tool.fc(x, w)

    if not b_shape:
        b = net_tool.bias_variable(b_shape)
        output = output + b

    return output


def dropout_layer(x, keep_prob=1.0):
    return net_tool.dropout(x, keep_prob)


def maxpool_layer(x):
    return net_tool.max_pool(x)