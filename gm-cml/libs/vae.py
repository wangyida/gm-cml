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

import os
import tensorflow as tf
import numpy as np
import sys
import csv
from libs.dataset_utils import create_input_pipeline
from libs.datasets import CELEB, MNIST
from libs.batch_norm import batch_norm
from libs import utils
from network import squeezenet

slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets import alexnet
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.contrib.slim.python.slim.nets import vgg

def VAE(input_shape=[None, 784],
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
    x_img = tf.placeholder(tf.float32, input_shape, 'x_img')
    x_obj = tf.placeholder(tf.float32, input_shape, 'x_obj')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    corrupt_rec = tf.placeholder(tf.float32, name='corrupt_rec')
    corrupt_cls = tf.placeholder(tf.float32, name='corrupt_cls')
    x_label = tf.placeholder(tf.int32, [None,1], 'x_label')

    # input of the reconstruction network
    # np.tanh(2) = 0.964
    current_input1 = utils.corrupt(x_img)*corrupt_rec + x_img*(1-corrupt_rec) \
                if (denoising and phase_train is not None) else x_img
    current_input1.set_shape(x_img.get_shape())
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
            # z_log_sigma = tf.matmul(tf.matmul(u, tf.diag(s)), tf.transpose(v))
            # end yidawang

            # Sample from noise distribution p(eps) ~ N(0, 1)
            epsilon = tf.random_normal(
                tf.stack([tf.shape(x_img)[0], n_code]))

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
    x_obj_flat = utils.flatten(x_obj)
    y_flat = utils.flatten(y)

    # l2 loss
    loss_x = tf.reduce_mean(
        tf.reduce_sum(tf.squared_difference(x_obj_flat, y_flat), 1))
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
        axis = list(range(len(x_img.get_shape())))
        mean1, variance1 = tf.nn.moments(x_obj, axis) \
                        if (phase_train is True) else tf.nn.moments(x_img, axis)
        mean2, variance2 = tf.nn.moments(y, axis)
        var_prob = variance2/variance1

        # Input of the classification network
        current_input2 = utils.corrupt(x_img)*corrupt_cls + \
                     x_img*(1-corrupt_cls) \
                     if (denoising and phase_train is True) else x_img
        current_input2.set_shape(x_img.get_shape())
        current_input2 = utils.to_tensor(current_input2) \
                    if convolutional else current_input2

        y_concat = tf.concat([current_input2, y], 3)
        with tf.variable_scope('deconv/concat'):
            shape = shapes[layer_i + 1]
            if convolutional:
        # Here we set the input of classification network is the twice of
        # the input of the reconstruction network
        # 112->224 for alexNet and 150->300 for inception v3 and v4
                y_concat, W = utils.deconv2d(x=y_concat,
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

        x_label_onehot = tf.squeeze(tf.one_hot(x_label, 13, 1, 0), [1])
        slim.losses.softmax_cross_entropy(predictions, x_label_onehot)
        cost_s = slim.losses.get_total_loss()
        # cost = tf.reduce_mean(cost + cost_s)
        acc = tf.nn.in_top_k(predictions, tf.squeeze(x_label, [1]), 1)
    else:
        predictions = tf.squeeze(tf.one_hot(x_label, 13, 1, 0), [1])
        x_label_onehot = tf.squeeze(tf.one_hot(x_label, 13, 1, 0), [1])
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
            'x_img': x_img,
            'x_obj': x_obj,
            'x_label': x_label,
            'x_label_onehot': x_label_onehot,
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


def train_vae(files_img,
              files_obj,
              input_shape,
              type_input='csv_path',
              learning_rate=0.0001,
              batch_size=100,
              n_epochs=50,
              n_examples=121,
              crop_shape=[128, 128, 3],
              crop_factor=0.8,
              n_filters=[75, 100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              denoising=True,
              convolutional=True,
              variational=True,
              softmax=False,
              classifier='alexnet_v2',
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=2500,
              save_step=100,
              ckpt_name="./vae.ckpt"):
    """General purpose training of a (Variational) (Convolutional) Autoencoder.

    Supply a list of file paths to images, and this will do everything else.

    Parameters
    ----------
    files : list of strings
    List of paths to images.
    input_shape : list
    Must define what the input image's shape is.
    type_input = str, optional
    Use csv files to train conditional VAE or just VAE.
    learning_rate : float, optional
    Learning rate.
    batch_size : int, optional
    Batch size.
    n_epochs : int, optional
    Number of epochs.
    n_examples : int, optional
    Number of example to use while demonstrating the current training
    iteration's reconstruction.  Creates a square montage, so make
    sure int(sqrt(n_examples))**2 = n_examples, e.g. 16, 25, 36, ... 100.
    crop_shape : list, optional
    Size to centrally crop the image to.
    crop_factor : float, optional
    Resize factor to apply before cropping.
    n_filters : list, optional
    Same as VAE's n_filters.
    n_hidden : int, optional
    Same as VAE's n_hidden.
    n_code : int, optional
    Same as VAE's n_code.
    convolutional : bool, optional
    Use convolution or not.
    variational : bool, optional
    Use variational layer or not.
    softmax : bool, optional
    Use the classification network or not.
    classifier : str, optional
    Network for classification.
    filter_sizes : list, optional
    Same as VAE's filter_sizes.
    dropout : bool, optional
    Use dropout or not
    keep_prob : float, optional
    Percent of keep for dropout.
    activation : function, optional
    Which activation function to use.
    img_step : int, optional
    How often to save training images showing the manifold and
    reconstruction.
    save_step : int, optional
    How often to save checkpoints.
    ckpt_name : str, optional
    Checkpoints will be named as this, e.g. 'model.ckpt'
    """
    tf.set_random_seed(1)
    seed=1
    batch_obj, batch_label_o = create_input_pipeline(
        files=files_obj,
        batch_size=batch_size,
        n_epochs=n_epochs,
        crop_shape=crop_shape,
        crop_factor=crop_factor,
        shape=input_shape,
        seed=seed,
        type_input=type_input)

    batch_img, batch_label_i = create_input_pipeline(
        files=files_img,
        batch_size=batch_size,
        n_epochs=n_epochs,
        crop_shape=crop_shape,
        crop_factor=crop_factor,
        shape=input_shape,
        seed=seed,
        type_input=type_input)

    if softmax:
        batch_imagenet, batch_imagenet_label = create_input_pipeline(
            files="../list_annotated_imagenet.csv",
            batch_size=batch_size,
            n_epochs=n_epochs,
            crop_shape=crop_shape,
            crop_factor=crop_factor,
            shape=input_shape,
            seed=seed,
            type_input=type_input)
        batch_pascal, batch_pascal_label = create_input_pipeline(
            files="../list_annotated_pascal.csv",
            batch_size=batch_size,
            n_epochs=n_epochs,
            crop_shape=crop_shape,
            crop_factor=crop_factor,
            shape=input_shape,
            seed=seed,
            type_input=type_input)
        batch_shapenet, batch_shapenet_label = create_input_pipeline(
            files="../list_annotated_img_test.csv",
            batch_size=batch_size,
            n_epochs=n_epochs,
            crop_shape=crop_shape,
            crop_factor=crop_factor,
            shape=input_shape,
            seed=seed,
            type_input=type_input)


    ae = VAE(input_shape=[None] + crop_shape,
             denoising=denoising,
             convolutional=convolutional,
             variational=variational,
             softmax=softmax,
             n_filters=n_filters,
             n_hidden=n_hidden,
             n_code=n_code,
             dropout=dropout,
             filter_sizes=filter_sizes,
             activation=activation,
             classifier=classifier)

    if (type_input == 'csv_path' or type_input == 'csv_feature'):
        with open(files_img,"r") as f:
            reader = csv.reader(f,delimiter = ",")
            data = list(reader)
            n_files = len(data)
    else:
        n_files = len(files_img)

    # Create a manifold of our inner most layer to show
    # example reconstructions.  This is one way to see
    # what the "embedding" or "latent space" of the encoder
    # is capable of encoding, though note that this is just
    # a random hyperplane within the latent space, and does not
    # encompass all possible embeddings.
    np.random.seed(1)
    zs = np.random.uniform(
        -1.0, 1.0, [4, n_code]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, n_examples)

    optimizer_vae = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(ae['cost_vae'])
    if softmax:
        # AlexNet for 0.01,
        # Iception v1 for 0.01
        # SqueezeNet for 0.01
        if classifier == 'inception_v3':
            lr = tf.train.exponential_decay(
                                    0.1,
                                    0,
                                    n_files/batch_size*20,
                                    0.16,
                                    staircase=True)
            optimizer_softmax = tf.train.RMSPropOptimizer(
                                    lr,
                                    decay=0.9,
                                    momentum=0.9,
                                    epsilon=0.1).minimize(ae['cost_s'])
        elif classifier == 'inception_v2':
            optimizer_softmax = tf.train.AdamOptimizer(
                                    learning_rate=0.01).minimize(ae['cost_s'])
        elif classifier == 'inception_v1':
            optimizer_softmax = tf.train.GradientDescentOptimizer(
                                    learning_rate=0.01).minimize(ae['cost_s'])
        elif (classifier == 'squeezenet') or (classifier == 'zigzagnet'):
            optimizer_softmax = tf.train.RMSPropOptimizer(
                                    0.04,
                                    decay=0.9,
                                    momentum=0.9,
                                    epsilon=0.1).minimize(ae['cost_s'])
        elif classifier == 'alexnet_v2':
            optimizer_softmax = tf.train.GradientDescentOptimizer(
                                    learning_rate=0.01).minimize(ae['cost_s'])
        else:
            optimizer_softmax = tf.train.GradientDescentOptimizer(
                                    learning_rate=0.001).minimize(ae['cost_s'])

    # We create a session to use the graph together with a GPU declaration.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()

    # Write a summary for Tensorboard
    train_writer = tf.summary.FileWriter('./summary', sess.graph)

    # initialize
    sess.run(tf.global_variables_initializer())

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)

    # Fit all training data
    t_i = 0
    step_i = 0
    batch_i = 0
    epoch_i = 0
    summary_i = 0
    cost = 0
    # Test samples of training data from ShapeNet
    test_xs_obj, test_xs_label = sess.run([batch_obj, batch_label_o])
    test_xs_obj /= 255.0
    utils.montage(test_xs_obj, 'train_obj.png')
    test_xs_img, test_xs_label = sess.run([batch_img, batch_label_i])
    test_xs_img /= 255.0
    utils.montage(test_xs_img, 'train_img.png')

    # Test samples of testing data from ImageNet
    test_imagenet_img, test_imagenet_label = \
                        sess.run([batch_imagenet, batch_imagenet_label])
    test_imagenet_img /= 255.0
    utils.montage(test_imagenet_img, 'test_imagenet_img.png')

    # Test samples of testing data from PASCAL 2012
    test_pascal_img, test_pascal_label = \
                        sess.run([batch_pascal, batch_pascal_label])
    test_pascal_img /= 255.0
    utils.montage(test_pascal_img, 'test_pascal_img.png')

    # Test samples of testing data from ShapeNet test data
    test_shapenet_img, test_shapenet_label = \
                        sess.run([batch_shapenet, batch_shapenet_label])
    test_shapenet_img /= 255.0
    utils.montage(test_shapenet_img, 'test_shapenet_img.png')
    try:
        while not coord.should_stop():
            batch_i += 1
            step_i += 1
            batch_xs_img, batch_xs_label = sess.run([batch_img, batch_label_i])
            batch_xs_img /= 255.0
            batch_xs_obj, batch_xs_label2 = sess.run([batch_obj, batch_label_o])
            batch_xs_obj /= 255.0
            #import pdb; pdb.set_trace()
            assert batch_xs_label.all() == batch_xs_label2.all()

            # Here we must set corrupt_rec and corrupt_cls as 0 to find a
            # proper ratio of variance to feed for variable var_prob.
            # We use tanh as non-linear function for ratio of Vars from
            # the reconstructed channels and original channels
            var_prob = sess.run(ae['var_prob'],
                        feed_dict={
                            ae['x_img']: test_xs_img,
                            ae['x_label']: test_xs_label,
                            ae['train']: True,
                            ae['keep_prob']: 1.0,
                            ae['corrupt_rec']: 0,
                            ae['corrupt_cls']: 0})

            # Here is a fast training process
            corrupt_rec = np.tanh(0.25*var_prob)
            corrupt_cls = np.tanh(1-np.tanh(2*var_prob))

            # Optimizing reconstruction network
            cost_vae = sess.run(
                [ae['cost_vae'], optimizer_vae],
                feed_dict={
                    ae['x_img']: batch_xs_img,
                    ae['x_obj']: batch_xs_obj,
                    ae['x_label']: batch_xs_label,
                    ae['train']: True,
                    ae['keep_prob']: keep_prob,
                    ae['corrupt_rec']: corrupt_rec,
                    ae['corrupt_cls']: corrupt_cls})[0]
            cost += cost_vae
            if softmax:

                # Optimizing classification network
                cost_s = sess.run(
                    [ae['cost_s'], optimizer_softmax],
                    feed_dict={
                        ae['x_img']: batch_xs_img,
                        ae['x_obj']: batch_xs_obj,
                        ae['x_label']: batch_xs_label,
                        ae['train']: True,
                        ae['keep_prob']: keep_prob,
                        ae['corrupt_rec']: corrupt_rec,
                        ae['corrupt_cls']: corrupt_cls})[0]
                cost += cost_s

            if step_i % img_step == 0:
                if variational:
                    # Plot example reconstructions from latent layer
                    recon = sess.run(
                        ae['y'], feed_dict={
                                    ae['z']: zs,
                                    ae['train']: False,
                                    ae['keep_prob']: 1.0,
                                    ae['corrupt_rec']: 0,
                                    ae['corrupt_cls']: 0})
                    utils.montage(recon.reshape([-1] + crop_shape),
                                  'manifold_%08d.png' % t_i)

                # Plot example reconstructions
                recon = sess.run(
                    ae['y'], feed_dict={
                                ae['x_img']: test_xs_img,
                                ae['train']: False,
                                ae['keep_prob']: 1.0,
                                ae['corrupt_rec']: 0,
                                ae['corrupt_cls']: 0})
                utils.montage(recon.reshape([-1] + crop_shape),
                              'recon_%08d.png' % t_i)
                """
                filters = sess.run(
                  ae['Ws'], feed_dict={
                              ae['x_img']: test_xs_img,
                              ae['train']: False,
                              ae['keep_prob']: 1.0,
                              ae['corrupt_rec']: 0,
                              ae['corrupt_cls']: 0})
                #for filter_element in filters:
                utils.montage_filters(filters[-1],
                            'filter_%08d.png' % t_i)
                """

                # Test on ImageNet samples
                with open('../list_annotated_imagenet.csv', 'r') as csvfile:
                    spamreader = csv.reader(csvfile)
                    rows = list(spamreader)
                    totalrows = len(rows)
                num_batches = np.int_(np.floor(totalrows/batch_size))
                accumulated_acc = 0
                for index_batch in range(1, num_batches+1):
                    test_image, test_label = sess.run(
                                        [batch_imagenet, batch_imagenet_label])
                    test_image /= 255.0
                    acc, z_codes, sm_codes = sess.run(
                                    [ae['acc'], ae['z'], ae['predictions']],
                                    feed_dict={
                                        ae['x_img']: test_image,
                                        ae['x_label']: test_label,
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0,
                                        ae['corrupt_rec']: 0,
                                        ae['corrupt_cls']: 0})
                    accumulated_acc += acc.tolist().count(True)/acc.size
                    if index_batch == 1:
                        z_imagenet = z_codes
                        sm_imagenet = sm_codes
                        labels_imagenet = test_label
                        # Plot example reconstructions
                        recon = sess.run(ae['y'],
                                    feed_dict={
                                        ae['x_img']: test_imagenet_img,
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0,
                                        ae['corrupt_rec']: 0,
                                        ae['corrupt_cls']: 0})
                        utils.montage(recon.reshape([-1] + crop_shape),
                                      'recon_imagenet_%08d.png' % t_i)
                    else:
                        z_imagenet = np.append(z_imagenet, z_codes, axis=0)
                        sm_imagenet = np.append(sm_imagenet, sm_codes, axis=0)
                        labels_imagenet = np.append(labels_imagenet, test_label,
                                                    axis=0)
                if variational:
                    mat = np.append(labels_imagenet, z_imagenet, axis=1)
                    np.save('feat_imagenet_z_%08d.npy' % t_i, mat)
                mat = np.append(labels_imagenet, sm_imagenet, axis=1)
                np.save('feat_imagenet_sm_%08d.npy' % t_i, mat)
                accumulated_acc /= num_batches
                print("Accuracy of ImageNet images= %.3f" % (accumulated_acc))

                # Test on PASCAL 2012 samples
                with open('../list_annotated_pascal.csv', 'r') as csvfile:
                    spamreader = csv.reader(csvfile)
                    rows = list(spamreader)
                    totalrows = len(rows)
                num_batches = np.int_(np.floor(totalrows/batch_size))
                accumulated_acc = 0
                for index_batch in range(1, num_batches+1):
                    test_image, test_label = sess.run(
                                    [batch_pascal, batch_pascal_label])
                    test_image /= 255.0
                    acc, z_codes, sm_codes = sess.run(
                                    [ae['acc'], ae['z'], ae['predictions']],
                                    feed_dict={
                                        ae['x_img']: test_image,
                                        ae['x_label']: test_label,
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0,
                                        ae['corrupt_rec']: 0,
                                        ae['corrupt_cls']: 0})
                    accumulated_acc += acc.tolist().count(True)/acc.size
                    if index_batch == 1:
                        z_pascal = z_codes
                        sm_pascal = sm_codes
                        labels_pascal = test_label
                        # Plot example reconstructions
                        recon = sess.run(ae['y'],
                                    feed_dict={
                                        ae['x_img']: test_pascal_img,
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0,
                                        ae['corrupt_rec']: 0,
                                        ae['corrupt_cls']: 0})
                        utils.montage(recon.reshape([-1] + crop_shape),
                                      'recon_pascal_%08d.png' % t_i)
                    else:
                        z_pascal = np.append(z_pascal, z_codes, axis=0)
                        sm_pascal = np.append(sm_pascal, sm_codes, axis=0)
                        labels_pascal = np.append(labels_pascal, test_label,
                                                  axis=0)
                if variational:
                    mat = np.append(labels_pascal, z_pascal, axis=1)
                    np.save('feat_pascal_z_%08d.npy' % t_i, mat)
                mat = np.append(labels_pascal, sm_pascal, axis=1)
                np.save('feat_pascal_sm_%08d.npy' % t_i, mat)
                accumulated_acc /= num_batches
                print("Accuracy of PASCAL images= %.3f" % (accumulated_acc))

                # Test on ShapeNet test samples
                with open('../list_annotated_img_test.csv', 'r') as csvfile:
                    spamreader = csv.reader(csvfile)
                    rows = list(spamreader)
                    totalrows = len(rows)
                num_batches = np.int_(np.floor(totalrows/batch_size))
                accumulated_acc = 0
                for index_batch in range(1, num_batches+1):
                    test_image, test_label = sess.run(
                                    [batch_shapenet, batch_shapenet_label])
                    test_image /= 255.0
                    acc, z_codes, sm_codes = sess.run(
                                    [ae['acc'], ae['z'], ae['predictions']],
                                    feed_dict={
                                        ae['x_img']: test_image,
                                        ae['x_label']: test_label,
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0,
                                        ae['corrupt_rec']: 0,
                                        ae['corrupt_cls']: 0})
                    accumulated_acc += acc.tolist().count(True)/acc.size
                    if index_batch == 1:
                        z_shapenet = z_codes
                        sm_shapenet = sm_codes
                        labels_shapenet = test_label
                        # Plot example reconstructions
                        recon = sess.run(ae['y'],
                                    feed_dict={
                                        ae['x_img']: test_shapenet_img,
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0,
                                        ae['corrupt_rec']: 0,
                                        ae['corrupt_cls']: 0})
                        utils.montage(recon.reshape([-1] + crop_shape),
                                      'recon_shapenet_%08d.png' % t_i)
                    else:
                        z_shapenet = np.append(z_shapenet, z_codes, axis=0)
                        sm_shapenet = np.append(sm_shapenet, sm_codes, axis=0)
                        labels_shapenet = np.append(labels_shapenet, test_label,
                                                    axis=0)
                if variational:
                    mat = np.append(labels_shapenet, z_shapenet, axis=1)
                    np.save('feat_shapenet_z_%08d.npy' % t_i, mat)
                mat = np.append(labels_shapenet, sm_shapenet, axis=1)
                np.save('feat_shapenet_sm_%08d.npy' % t_i, mat)
                accumulated_acc /= num_batches
                print("Accuracy of ShapeNet images= %.3f" % (accumulated_acc))

                t_i += 1

            if step_i % save_step == 0:

                # Save the variables to disk.
                saver.save(sess, "./" + ckpt_name,
                           global_step=step_i,
                           write_meta_graph=False)
                if softmax:
                    acc = sess.run(ae['acc'],
                                feed_dict={
                                    ae['x_img']: test_xs_img,
                                    ae['x_label']: test_xs_label,
                                    ae['train']: False,
                                    ae['keep_prob']: 1.0,
                                    ae['corrupt_rec']: 0,
                                    ae['corrupt_cls']: 0})

                    print("epoch %d: VAE = %d, SM = %.3f, Acc = %.3f, R_Var = %.3f, Cpt_R = %.3f, Cpt_C = %.3f" %
                          (epoch_i,
                          cost_vae,
                          cost_s,
                          acc.tolist().count(True)/acc.size,
                          var_prob,
                          corrupt_rec,
                          corrupt_cls))

                    # Summary recording to Tensorboard
                    summary = sess.run(ae['merged'],
                                feed_dict={
                                    ae['x_img']: batch_xs_img,
                                    ae['x_obj']: batch_xs_obj,
                                    ae['x_label']: batch_xs_label,
                                    ae['train']: False,
                                    ae['keep_prob']: keep_prob,
                                    ae['corrupt_rec']: corrupt_rec,
                                    ae['corrupt_cls']: corrupt_cls})

                    summary_i += 1
                    train_writer.add_summary(summary, summary_i)
                else:
                    print("VAE loss = %d" % cost_vae)

            if batch_i > (n_files/batch_size):
                batch_i = 0
                epoch_i += 1

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        coord.request_stop()

    # Wait until all threads have finished.
    coord.join(threads)

    # Clean up the session.
    sess.close()

if __name__ == '__main__':
    test_celeb()
