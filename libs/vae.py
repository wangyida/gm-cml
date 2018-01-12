"""
TensorFlow implementation for Adaptive noise assisted Conjugating Generative
Model based on Conditional Variational Autoencoder, this project is
implemented based on VAE code pool of Parag K. Mital from Kadenze course
on Tensorflow and modified by Yida Wang for the paper of
'Conjugating Generative Model for Object Recognition Based on 3D Models'.

There is also implementation of ZigzagNet and SqueezeNet for compact deep
learninig for classifcation.

Copyright reserved for Yida Wang from BUPT.
"""

import tensorflow as tf
from libs.batch_norm import batch_norm
from libs import utils
from network import squeezenet

slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets import alexnet
from tensorflow.contrib.slim.python.slim.nets import inception


def VAE(input_shape=[None, 784],
        output_shape=[None, 784],
        n_filters=[64, 64, 64],
        filter_sizes=[4, 4, 4],
        n_hidden=32,
        n_code=2,
        activation=tf.nn.tanh,
        dropout=False,
        denoising=False,
        convolutional=False,
        variational=False,
        softmax=False,
        classifier='alexnet_v2'):
    """(Variational) (Convolutional) (Denoising) Autoencoder.

    Uses tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Shape of the input to the network. e.g. for MNIST: [None, 784].
    n_filters : list, optional
        Number of filters for each layer.
        If convolutional=True, this refers to the total number of output
        filters to create for each layer, with each layer's number of output
        filters as a list.
        If convolutional=False, then this refers to the total number of neurons
        for each layer in a fully connected network.
    filter_sizes : list, optional
        Only applied when convolutional=True.  This refers to the ksize (height
        and width) of each convolutional layer.
    n_hidden : int, optional
        Only applied when variational=True.  This refers to the first fully
        connected layer prior to the variational embedding, directly after
        the encoding.  After the variational embedding, another fully connected
        layer is created with the same size prior to decoding.  Set to 0 to
        not use an additional hidden layer.
    n_code : int, optional
        Only applied when variational=True.  This refers to the number of
        latent Gaussians to sample for creating the inner most encoding.
    activation : function, optional
        Activation function to apply to each layer, e.g. tf.nn.relu
    dropout : bool, optional
        Whether or not to apply dropout.  If using dropout, you must feed a
        value for 'keep_prob', as returned in the dictionary.  1.0 means no
        dropout is used.  0.0 means every connection is dropped.  Sensible
        values are between 0.5-0.8.
    denoising : bool, optional
        Whether or not to apply denoising.  If using denoising, you must feed a
        value for 'corrupt_rec', as returned in the dictionary.  1.0 means no
        corruption is used.  0.0 means every feature is corrupted.  Sensible
        values are between 0.5-0.8.
    convolutional : bool, optional
        Whether or not to use a convolutional network or else a fully connected
        network will be created.  This effects the n_filters parameter's
        meaning.
    variational : bool, optional
        Whether or not to create a variational embedding layer.  This will
        create a fully connected layer after the encoding, if `n_hidden` is
        greater than 0, then will create a multivariate gaussian sampling
        layer, then another fully connected layer.  The size of the fully
        connected layers are determined by `n_hidden`, and the size of the
        sampling layer is determined by `n_code`.

    Returns
    -------
    model : dict
        {
            'cost': Tensor to optimize.
            'Ws': All weights of the encoder.
            'x': Input Placeholder
            'z': Inner most encoding Tensor (latent features)
            'y': Reconstruction of the Decoder
            'keep_prob': Amount to keep when using Dropout
            'corrupt_rec': Amount to corrupt when using Denoising
            'train': Set to True when training/Applies to Batch Normalization.
        }
    """
    # network input / placeholders for train (bn) and dropout
    x = tf.placeholder(tf.float32, input_shape, 'x')
    t = tf.placeholder(tf.float32, output_shape, 't')
    label = tf.placeholder(tf.int32, [None], 'label')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    corrupt_rec = tf.placeholder(tf.float32, name='corrupt_rec')
    corrupt_cls = tf.placeholder(tf.float32, name='corrupt_cls')

    # input of the reconstruction network
    # np.tanh(2) = 0.964
    current_input1 = utils.corrupt(x)*corrupt_rec + x*(1-corrupt_rec) \
        if (denoising and phase_train is not None) else x
    current_input1.set_shape(x.get_shape())
    # 2d -> 4d if convolution
    current_input1 = utils.to_tensor(current_input1) \
        if convolutional else current_input1

    Ws = []
    shapes = []

    # Build the encoder
    for layer_i, n_output in enumerate(n_filters):
        with tf.variable_scope('encoder/{}'.format(layer_i)):
            shapes.append(current_input1.get_shape().as_list())
            if convolutional:
                h, W = utils.conv2d(x=current_input1,
                                    n_output=n_output,
                                    k_h=filter_sizes[layer_i],
                                    k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input1,
                                    n_output=n_output)
            h = activation(batch_norm(h, phase_train, 'bn' + str(layer_i)))
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            Ws.append(W)
            current_input1 = h

    shapes.append(current_input1.get_shape().as_list())

    with tf.variable_scope('variational'):
        if variational:
            dims = current_input1.get_shape().as_list()
            flattened = utils.flatten(current_input1)

            if n_hidden:
                h = utils.linear(flattened, n_hidden, name='W_fc')[0]
                h = activation(batch_norm(h, phase_train, 'fc/bn'))
                if dropout:
                    h = tf.nn.dropout(h, keep_prob)
            else:
                h = flattened

            z_mu = utils.linear(h, n_code, name='mu')[0]
            z_log_sigma = 0.5 * utils.linear(h, n_code, name='log_sigma')[0]
            # modified by yidawang
            # s, u, v = tf.svd(z_log_sigma)
            # z_log_sigma = tf.matmul(
            #        tf.matmul(u, tf.diag(s)), tf.transpose(v))
            # end yidawang

            # Sample from noise distribution p(eps) ~ N(0, 1)
            epsilon = tf.random_normal(
                tf.stack([tf.shape(x)[0], n_code]))

            # Sample from posterior
            z = z_mu + tf.multiply(epsilon, tf.exp(z_log_sigma))

            if n_hidden:
                h = utils.linear(z, n_hidden, name='fc_t')[0]
                h = activation(batch_norm(h, phase_train, 'fc_t/bn'))
                if dropout:
                    h = tf.nn.dropout(h, keep_prob)
            else:
                h = z

            size = dims[1] * dims[2] * dims[3] if convolutional else dims[1]
            h = utils.linear(h, size, name='fc_t2')[0]
            current_input1 = activation(batch_norm(h, phase_train, 'fc_t2/bn'))
            if dropout:
                current_input1 = tf.nn.dropout(current_input1, keep_prob)

            if convolutional:
                current_input1 = tf.reshape(
                    current_input1, tf.stack([
                        tf.shape(current_input1)[0],
                        dims[1],
                        dims[2],
                        dims[3]]))
        else:
            z = current_input1

    shapes.reverse()
    n_filters.reverse()
    Ws.reverse()

    n_filters += [input_shape[-1]]

    # %%
    # Decoding layers
    for layer_i, n_output in enumerate(n_filters[1:]):
        with tf.variable_scope('decoder/{}'.format(layer_i)):
            shape = shapes[layer_i + 1]
            if convolutional:
                h, W = utils.deconv2d(x=current_input1,
                                      n_output_h=shape[1],
                                      n_output_w=shape[2],
                                      n_output_ch=shape[3],
                                      n_input_ch=shapes[layer_i][3],
                                      k_h=filter_sizes[layer_i],
                                      k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input1,
                                    n_output=n_output)
            h = activation(batch_norm(h, phase_train, 'dec/bn' + str(layer_i)))
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            current_input1 = h

    y = current_input1
    t_flat = utils.flatten(t)
    y_flat = utils.flatten(y)

    # l2 loss
    loss_x = tf.reduce_mean(
        tf.reduce_sum(tf.squared_difference(t_flat, y_flat), 1))
    loss_z = 0

    if variational:
        # Variational lower bound, kl-divergence
        loss_z = tf.reduce_mean(-0.5 * tf.reduce_sum(
            1.0 + 2.0 * z_log_sigma -
            tf.square(z_mu) - tf.exp(2.0 * z_log_sigma), 1))

        # Add l2 loss
        cost_vae = tf.reduce_mean(loss_x + loss_z)
    else:
        # Just optimize l2 loss
        cost_vae = tf.reduce_mean(loss_x)

    # Alexnet for clasification based on softmax using TensorFlow slim
    if softmax:
        axis = list(range(len(x.get_shape())))
        mean1, variance1 = tf.nn.moments(t, axis) \
            if (phase_train is True) else tf.nn.moments(x, axis)
        mean2, variance2 = tf.nn.moments(y, axis)
        var_prob = variance2/variance1

        # Input of the classification network
        current_input2 = utils.corrupt(x)*corrupt_cls + \
            x*(1-corrupt_cls) \
            if (denoising and phase_train is True) else x
        current_input2.set_shape(x.get_shape())
        current_input2 = utils.to_tensor(current_input2) \
            if convolutional else current_input2

        y_concat = tf.concat([current_input2, y], 3)
        with tf.variable_scope('deconv/concat'):
            shape = shapes[layer_i + 1]
            if convolutional:
                # Here we set the input of classification network is
                # the twice of
                # the input of the reconstruction network
                # 112->224 for alexNet and 150->300 for inception v3 and v4
                y_concat, W = utils.deconv2d(
                        x=y_concat,
                        n_output_h=y_concat.get_shape()[1]*2,
                        n_output_w=y_concat.get_shape()[1]*2,
                        n_output_ch=y_concat.get_shape()[3],
                        n_input_ch=y_concat.get_shape()[3],
                        k_h=3,
                        k_w=3)
                Ws.append(W)

        # The following are optional networks for classification network
        if classifier == 'squeezenet':
            predictions, net = squeezenet.squeezenet(
                        y_concat, num_classes=13)
        if classifier == 'zigzagnet':
            predictions, net = squeezenet.zigzagnet(
                        y_concat, num_classes=13)
        elif classifier == 'alexnet_v2':
            predictions, end_points = alexnet.alexnet_v2(
                        y_concat, num_classes=13)
        elif classifier == 'inception_v1':
            predictions, end_points = inception.inception_v1(
                        y_concat, num_classes=13)
        elif classifier == 'inception_v2':
            predictions, end_points = inception.inception_v2(
                        y_concat, num_classes=13)
        elif classifier == 'inception_v3':
            predictions, end_points = inception.inception_v3(
                        y_concat, num_classes=13)

        label_onehot = tf.one_hot(label, 13, 1, 0)
        slim.losses.softmax_cross_entropy(predictions, label_onehot)
        cost_s = slim.losses.get_total_loss()
        # cost = tf.reduce_mean(cost + cost_s)
        acc = tf.nn.in_top_k(predictions, label, 1)
    else:
        predictions = tf.one_hot(label, 13, 1, 0)
        label_onehot = tf.one_hot(label, 13, 1, 0)
        cost_s = 0
        acc = 0
    # Using Summaries for Tensorboard
    tf.summary.scalar('cost_vae', cost_vae)
    tf.summary.scalar('cost_s', cost_s)
    tf.summary.scalar('loss_x', loss_x)
    tf.summary.scalar('loss_z', loss_z)
    tf.summary.scalar('corrupt_rec', corrupt_rec)
    tf.summary.scalar('corrupt_cls', corrupt_cls)
    tf.summary.scalar('var_prob', var_prob)
    merged = tf.summary.merge_all()

    return {'cost_vae': cost_vae,
            'cost_s': cost_s,
            'loss_x': loss_x,
            'loss_z': loss_z,
            'Ws': Ws,
            'x': x,
            't': t,
            'label': label,
            'label_onehot': label_onehot,
            'predictions': predictions,
            'z': z,
            'y': y,
            'acc': acc,
            'keep_prob': keep_prob,
            'corrupt_rec': corrupt_rec,
            'corrupt_cls': corrupt_cls,
            'var_prob': var_prob,
            'train': phase_train,
            'merged': merged}

