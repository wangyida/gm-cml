import os
import tensorflow as tf

def relu_layer(layer_name, layer_input, b):
    relu = tf.nn.relu(layer_input + b)
    return relu


def pool_layer(layer_name, layer_input, pooling_type='max'):
    if pooling_type == 'avg':
        pool = tf.nn.avg_pool(layer_input, ksize=[1, 13, 13, 1],
                              strides=[1, 1, 1, 1], padding='VALID')
    elif pooling_type == 'max':
        pool = tf.nn.max_pool(layer_input, ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
    return pool


def fire_module(layer_name, layer_input, s1x1, e1x1, e3x3):
    fire = {}

    shape = layer_input.get_shape()

    s1_weight = weight_variable([1, 1, int(shape[3]), s1x1], layer_name + '_s1')
    e1_weight = weight_variable([1, 1, s1x1, e1x1], layer_name + '_e1')
    e3_weight = weight_variable([1, 1, s1x1, e3x3], layer_name + '_e3')

    fire['s1'] = tf.nn.conv2d(layer_input, s1_weight, strides=[1,1,1,1], padding='SAME')

    fire['relu1'] = relu_layer(layer_name + '_relu1', fire['s1'], b=bias_variable([s1x1]))

    fire['e1'] = tf.nn.conv2d(fire['relu1'], e1_weight, strides=[1,1,1,1], padding='SAME')
    fire['e3'] = tf.nn.conv2d(fire['relu1'], e3_weight, strides=[1,1,1,1], padding='SAME')
    fire['concat'] = tf.concat_v2([fire['e1'], fire['e3']], 3)

    fire['relu2'] = relu_layer(layer_name + '_relu2', fire['concat'], b=bias_variable([e1x1 + e3x3]))

    return fire['relu2']

def zigzag_module(layer_name, layer_input, s1x1, e1x1, e3x3):
    zigzag = {}

    shape = layer_input.get_shape()

    s1_weight = weight_variable([1, 1, int(shape[3]), s1x1], layer_name + '_s1')
    e1_weight = weight_variable([1, 1, s1x1, e1x1], layer_name + '_e1')
    e3_weight = weight_variable([1, 1, s1x1, e3x3], layer_name + '_e3')
    b1_weight = weight_variable([1, 1, int(shape[3]), e1x1+e3x3], layer_name + '_b1')
    b2_weight = weight_variable([1, 1, 2*(e1x1+e3x3), e1x1+e3x3], layer_name + '_b2')

    zigzag['s1'] = tf.nn.conv2d(layer_input, s1_weight, strides=[1,1,1,1], padding='SAME')
    zigzag['relu1'] = relu_layer(layer_name + '_relu1', zigzag['s1'], b=bias_variable([s1x1]))

    zigzag['e1'] = tf.nn.conv2d(zigzag['relu1'], e1_weight, strides=[1,1,1,1], padding='SAME')
    zigzag['e3'] = tf.nn.conv2d(zigzag['relu1'], e3_weight, strides=[1,1,1,1], padding='SAME')
    zigzag['concat1'] = tf.concat_v2([zigzag['e1'], zigzag['e3']], 3)
    zigzag['b1'] = tf.nn.conv2d(layer_input, b1_weight, strides=[1,1,1,1], padding='SAME')
    zigzag['concat2'] = tf.concat_v2([zigzag['concat1'], zigzag['b1']], 3)
    zigzag['relu2'] = relu_layer(layer_name + '_relu2', zigzag['concat2'],
                               b=bias_variable([2*(e1x1+e3x3)]))

    zigzag['b2'] = tf.nn.conv2d(zigzag['relu2'], b2_weight, strides=[1,1,1,1], padding='SAME')
    zigzag['relu3'] = relu_layer(layer_name + '_relu3', zigzag['b2'],
                               b=bias_variable([e1x1+e3x3]))

    return zigzag['relu3']

def weight_variable(shape, name=None, init='xavier'):
    if init == 'xavier':
        initial = tf.get_variable('W' + name, shape, initializer=tf.contrib.layers.xavier_initializer())
    else:
        initial = tf.Variable(tf.truncated_normal(shape, stddev=0.01))

    return initial


def weight_xavier(shape, num_in, num_out):
    low = -4 * np.sqrt(6.0 / (num_in + num_out))  # {sigmoid:4, tanh:1}
    high = 4 * np.sqrt(6.0 / (num_in + num_out))
    return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def squeezenet(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='squeezenet'):
    net = {}
    # input - placeholder: add batch size to 1st layer later
    net['input'] = inputs
    # conv1_1
    net['conv1'] = tf.nn.conv2d(net['input'],
                                weight_variable([3, 3, 6, 64],name='conv1'),
                                strides=[1, 2, 2, 1], padding='SAME')

    net['relu1'] = relu_layer('relu1', net['conv1'], b=bias_variable([64]))

    net['pool1'] = pool_layer('pool1', net['relu1'])

    # fire2

    net['fire2'] = fire_module('fire2', net['pool1'], 16, 64, 64)
    net['fire3'] = fire_module('fire3', net['fire2'], 16, 64, 64)
    # maxpool
    net['pool3'] = pool_layer('pool3', net['fire3'])

    net['fire4'] = fire_module('fire4', net['pool3'], 32, 128, 128)
    net['fire5'] = fire_module('fire5', net['fire4'], 32, 128, 128)
    net['pool5'] = pool_layer('pool5', net['fire5'])

    net['fire6'] = fire_module('fire6', net['pool5'], 48, 192, 192)
    net['fire7'] = fire_module('fire7', net['fire6'], 48, 192, 192)
    net['fire8'] = fire_module('fire8', net['fire7'], 64, 256, 256)
    net['fire9'] = fire_module('fire9', net['fire8'], 64, 256, 256)
    # 50% dropout
    if is_training:
        net['dropout9'] = tf.nn.dropout(net['fire9'], dropout_keep_prob)
    else:
        net['dropout9'] = tf.nn.dropout(net['fire9'], 1)

    net['conv10'] = tf.nn.conv2d(net['dropout9'],
                                weight_variable([1, 1, 512, num_classes], name='conv10', init='gauss'), strides=[1,1,1,1], padding='SAME')
    net['relu10'] = relu_layer('relu10', net['conv10'], b=bias_variable([num_classes]))

    net['pool10'] = pool_layer('pool10', net['conv10'], pooling_type='avg')

    if spatial_squeeze:
      net['pool10'] = tf.squeeze(net['pool10'], [1, 2], name='output/squeezed')

    return net['pool10'], net

def zigzagnet(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='zigzagnet'):
    net = {}
    # input - placeholder: add batch size to 1st layer later
    net['input'] = inputs
    # conv1_1
    net['conv1'] = tf.nn.conv2d(net['input'],
                                weight_variable([3, 3, 6, 64],name='conv1'),
                                strides=[1, 2, 2, 1], padding='SAME')

    net['relu1'] = relu_layer('relu1', net['conv1'], b=bias_variable([64]))

    net['pool1'] = pool_layer('pool1', net['relu1'])

    # zigzag2

    net['zigzag2'] = zigzag_module('zigzag2', net['pool1'], 16, 64, 64)
    net['zigzag3'] = zigzag_module('zigzag3', net['zigzag2'], 16, 64, 64)
    # maxpool
    net['pool3'] = pool_layer('pool3', net['zigzag3'])

    net['zigzag4'] = zigzag_module('zigzag4', net['pool3'], 32, 128, 128)
    net['zigzag5'] = zigzag_module('zigzag5', net['zigzag4'], 32, 128, 128)
    net['pool5'] = pool_layer('pool5', net['zigzag5'])

    net['zigzag6'] = zigzag_module('zigzag6', net['pool5'], 48, 192, 192)
    net['zigzag7'] = zigzag_module('zigzag7', net['zigzag6'], 48, 192, 192)
    net['zigzag8'] = zigzag_module('zigzag8', net['zigzag7'], 64, 256, 256)
    net['zigzag9'] = zigzag_module('zigzag9', net['zigzag8'], 64, 256, 256)
    # 50% dropout
    if is_training:
        net['dropout9'] = tf.nn.dropout(net['zigzag9'], dropout_keep_prob)
    else:
        net['dropout9'] = tf.nn.dropout(net['zigzag9'], 1)

    net['conv10'] = tf.nn.conv2d(net['dropout9'],
                                weight_variable([1, 1, 512, num_classes], name='conv10', init='gauss'), strides=[1,1,1,1], padding='SAME')
    net['relu10'] = relu_layer('relu10', net['conv10'], b=bias_variable([num_classes]))

    net['pool10'] = pool_layer('pool10', net['conv10'], pooling_type='avg')

    if spatial_squeeze:
      net['pool10'] = tf.squeeze(net['pool10'], [1, 2], name='output/squeezed')

    return net['pool10'], net
