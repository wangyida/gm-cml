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

import matplotlib
matplotlib.use('Agg')
import os
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from pca import pca
from libs.dataset_utils import create_input_pipeline
from libs.vae import VAE
from libs import utils

slim = tf.contrib.slim


def train_vae(files,
              input_shape=[None, 784],
              output_shape=[None, 784],
              learning_rate=0.0001,
              batch_size=128,
              n_epochs=50,
              crop_shape=[64, 64],
              crop_factor=0.8,
              n_filters=[100, 100, 100, 100],
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
              img_step=1000,
              save_step=2500,
              output_path="result",
              ckpt_name="vae.ckpt"):
    """General purpose training of a (Variational) (Convolutional) Autoencoder.

    Supply a list of file paths to images, and this will do everything else.

    Parameters
    ----------
    files : list of strings
    List of paths to images.
    input_shape : list
    Must define what the input image's shape is.
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

    batch_train = create_input_pipeline(
        files=files,
        batch_size=batch_size,
        n_epochs=n_epochs,
        crop_shape=crop_shape,
        crop_factor=crop_factor,
        input_shape=input_shape,
        output_shape=output_shape)

    if softmax:
        batch_imagenet = create_input_pipeline(
            files="./list_annotated_imagenet.csv",
            batch_size=batch_size,
            n_epochs=n_epochs,
            crop_shape=crop_shape,
            crop_factor=crop_factor,
            input_shape=input_shape,
            output_shape=output_shape)
        batch_pascal = create_input_pipeline(
            files="./list_annotated_pascal.csv",
            batch_size=batch_size,
            n_epochs=n_epochs,
            crop_shape=crop_shape,
            crop_factor=crop_factor,
            input_shape=input_shape,
            output_shape=output_shape)
        batch_shapenet = create_input_pipeline(
            files="./list_annotated_img_test.csv",
            batch_size=batch_size,
            n_epochs=n_epochs,
            crop_shape=crop_shape,
            crop_factor=crop_factor,
            input_shape=input_shape,
            output_shape=output_shape)

    ae = VAE(input_shape=[None] + crop_shape + [input_shape[-1]],
             output_shape=[None] + crop_shape + [output_shape[-1]],
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

    with open(files, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = list(reader)
        n_files = len(data)

    # Create a manifold of our inner most layer to show
    # example reconstructions.  This is one way to see
    # what the "embedding" or "latent space" of the encoder
    # is capable of encoding, though note that this is just
    # a random hyperplane within the latent space, and does not
    # encompass all possible embeddings.
    np.random.seed(1)
    zs = np.random.uniform(
        -1.0, 1.0, [4, n_code]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, 6)

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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./summary', sess.graph)

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if (
            os.path.exists(output_path + '/' + ckpt_name + '.index') or
            os.path.exists(ckpt_name)
       ):
        saver.restore(sess, output_path + '/' + ckpt_name)
        print("Model restored")

    # Fit all training data
    t_i = 0
    step_i = 0
    batch_i = 0
    epoch_i = 0
    summary_i = 0
    cost = 0
    # Test samples of training data from ShapeNet
    test_xs_img, test_xs_obj, test_xs_label = sess.run(batch_train)
    test_xs_img /= 255.0
    test_xs_obj /= 255.0
    utils.montage(test_xs_img, output_path + '/train_img.png')
    utils.montage(test_xs_obj, output_path + '/train_obj.png')

    # Test samples of testing data from ImageNet
    test_imagenet_img, _, test_imagenet_label = sess.run(batch_imagenet)
    test_imagenet_img /= 255.0
    utils.montage(test_imagenet_img, output_path + '/test_imagenet_img.png')

    # Test samples of testing data from PASCAL 2012
    test_pascal_img, _, test_pascal_label = sess.run(batch_pascal)
    test_pascal_img /= 255.0
    utils.montage(test_pascal_img, output_path + '/test_pascal_img.png')

    # Test samples of testing data from ShapeNet test data
    test_shapenet_img, _, test_shapenet_label = sess.run(batch_shapenet)
    test_shapenet_img /= 255.0
    utils.montage(test_shapenet_img, output_path + '/test_shapenet_img.png')
    try:
        while not coord.should_stop():
            batch_i += 1
            step_i += 1
            batch_xs_img, batch_xs_obj, batch_xs_label = sess.run(batch_train)
            batch_xs_img /= 255.0
            batch_xs_obj /= 255.0

            # Here we must set corrupt_rec and corrupt_cls as 0 to find a
            # proper ratio of variance to feed for variable var_prob.
            # We use tanh as non-linear function for ratio of Vars from
            # the reconstructed channels and original channels
            var_prob = sess.run(
                    ae['var_prob'],
                    feed_dict={
                        ae['x']: test_xs_img,
                        ae['label']: test_xs_label[:, 0],
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
                    ae['x']: batch_xs_img,
                    ae['t']: batch_xs_obj,
                    ae['label']: batch_xs_label[:, 0],
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
                        ae['x']: batch_xs_img,
                        ae['t']: batch_xs_obj,
                        ae['label']: batch_xs_label[:, 0],
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
                                  output_path + '/manifold_%08d.png' % t_i)

                # Plot example reconstructions
                recon = sess.run(
                    ae['y'], feed_dict={
                                ae['x']: test_xs_img,
                                ae['train']: False,
                                ae['keep_prob']: 1.0,
                                ae['corrupt_rec']: 0,
                                ae['corrupt_cls']: 0})
                utils.montage(recon.reshape([-1] + crop_shape),
                              output_path + '/recon_%08d.png' % t_i)
                """
                filters = sess.run(
                  ae['Ws'], feed_dict={
                              ae['x']: test_xs_img,
                              ae['train']: False,
                              ae['keep_prob']: 1.0,
                              ae['corrupt_rec']: 0,
                              ae['corrupt_cls']: 0})
                #for filter_element in filters:
                utils.montage_filters(filters[-1],
                                output_path + '/filter_%08d.png' % t_i)
                """

                # Test on ImageNet samples
                with open('./list_annotated_imagenet.csv', 'r') as csvfile:
                    spamreader = csv.reader(csvfile)
                    rows = list(spamreader)
                    totalrows = len(rows)
                num_batches = np.int_(np.floor(totalrows/batch_size))
                accumulated_acc = 0
                for index_batch in range(1, num_batches+1):
                    test_image, _, test_label = sess.run(batch_imagenet)
                    test_image /= 255.0
                    acc, z_codes, sm_codes = sess.run(
                                    [ae['acc'], ae['z'], ae['predictions']],
                                    feed_dict={
                                        ae['x']: test_image,
                                        ae['label']: test_label[:, 0],
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
                        recon = sess.run(
                                ae['y'],
                                feed_dict={
                                    ae['x']: test_imagenet_img,
                                    ae['train']: False,
                                    ae['keep_prob']: 1.0,
                                    ae['corrupt_rec']: 0,
                                    ae['corrupt_cls']: 0})
                        utils.montage(recon.reshape([-1] + crop_shape),
                                      output_path + '/recon_imagenet_%08d.png' % t_i)
                    else:
                        z_imagenet = np.append(z_imagenet, z_codes, axis=0)
                        sm_imagenet = np.append(sm_imagenet, sm_codes, axis=0)
                        labels_imagenet = np.append(
                                labels_imagenet,
                                test_label,
                                axis=0)
                accumulated_acc /= num_batches
                print("Accuracy of ImageNet images= %.3f" % (accumulated_acc))

                fig = plt.figure()
                z_viz, V = pca(z_imagenet, dim_remain=2)
                ax = fig.add_subplot(121)
                # ax.set_aspect('equal')
                ax.scatter(
                        z_viz[:, 0],
                        z_viz[:, 1],
                        c=labels_imagenet[:, 0],
                        alpha=0.4,
                        cmap='gist_rainbow')
                sm_viz, V = pca(sm_imagenet, dim_remain=2)
                ax = fig.add_subplot(122)
                # ax.set_aspect('equal')
                ax.scatter(
                        sm_viz[:, 0],
                        sm_viz[:, 1],
                        c=labels_imagenet[:, 0],
                        alpha=0.4,
                        cmap='gist_rainbow')

                fig.savefig(output_path + '/z_feat_imagenet.png', transparent=True)
                plt.clf()

                # Test on PASCAL 2012 samples
                with open('./list_annotated_pascal.csv', 'r') as csvfile:
                    spamreader = csv.reader(csvfile)
                    rows = list(spamreader)
                    totalrows = len(rows)
                num_batches = np.int_(np.floor(totalrows/batch_size))
                accumulated_acc = 0
                for index_batch in range(1, num_batches+1):
                    test_image, _, test_label = sess.run(batch_pascal)
                    test_image /= 255.0
                    acc, z_codes, sm_codes = sess.run(
                                    [ae['acc'], ae['z'], ae['predictions']],
                                    feed_dict={
                                        ae['x']: test_image,
                                        ae['label']: test_label[:, 0],
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
                        recon = sess.run(
                                ae['y'],
                                feed_dict={
                                    ae['x']: test_pascal_img,
                                    ae['train']: False,
                                    ae['keep_prob']: 1.0,
                                    ae['corrupt_rec']: 0,
                                    ae['corrupt_cls']: 0})
                        utils.montage(recon.reshape([-1] + crop_shape),
                                      output_path + '/recon_pascal_%08d.png' % t_i)
                    else:
                        z_pascal = np.append(z_pascal, z_codes, axis=0)
                        sm_pascal = np.append(sm_pascal, sm_codes, axis=0)
                        labels_pascal = np.append(labels_pascal, test_label,
                                                  axis=0)
                accumulated_acc /= num_batches
                print("Accuracy of PASCAL images= %.3f" % (accumulated_acc))

                fig = plt.figure()
                z_viz, V = pca(z_pascal, dim_remain=2)
                ax = fig.add_subplot(121)
                # ax.set_aspect('equal')
                ax.scatter(
                        z_viz[:, 0],
                        z_viz[:, 1],
                        c=labels_pascal[:, 0],
                        alpha=0.4,
                        cmap='gist_rainbow')
                sm_viz, V = pca(sm_pascal, dim_remain=2)
                ax = fig.add_subplot(122)
                # ax.set_aspect('equal')
                ax.scatter(
                        sm_viz[:, 0],
                        sm_viz[:, 1],
                        c=labels_pascal[:, 0],
                        alpha=0.4,
                        cmap='gist_rainbow')

                fig.savefig(output_path + '/z_feat_pascal.png', transparent=True)
                plt.clf()

                # Test on ShapeNet test samples
                with open('./list_annotated_img_test.csv', 'r') as csvfile:
                    spamreader = csv.reader(csvfile)
                    rows = list(spamreader)
                    totalrows = len(rows)
                num_batches = np.int_(np.floor(totalrows/batch_size))
                accumulated_acc = 0
                for index_batch in range(1, num_batches+1):
                    test_image, _, test_label = sess.run(batch_shapenet)
                    test_image /= 255.0
                    acc, z_codes, sm_codes = sess.run(
                                    [ae['acc'], ae['z'], ae['predictions']],
                                    feed_dict={
                                        ae['x']: test_image,
                                        ae['label']: test_label[:, 0],
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
                        recon = sess.run(
                                ae['y'],
                                feed_dict={
                                        ae['x']: test_shapenet_img,
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0,
                                        ae['corrupt_rec']: 0,
                                        ae['corrupt_cls']: 0})
                        utils.montage(recon.reshape([-1] + crop_shape),
                                      output_path + '/recon_shapenet_%08d.png' % t_i)
                    else:
                        z_shapenet = np.append(z_shapenet, z_codes, axis=0)
                        sm_shapenet = np.append(sm_shapenet, sm_codes, axis=0)
                        labels_shapenet = np.append(
                                labels_shapenet,
                                test_label,
                                axis=0)
                accumulated_acc /= num_batches
                print("Accuracy of ShapeNet images= %.3f" % (accumulated_acc))

                fig = plt.figure()
                z_viz, V = pca(z_shapenet, dim_remain=2)
                ax = fig.add_subplot(121)
                # ax.set_aspect('equal')
                ax.scatter(
                        z_viz[:, 0],
                        z_viz[:, 1],
                        c=labels_shapenet[:, 0],
                        alpha=0.4,
                        cmap='gist_rainbow')
                sm_viz, V = pca(sm_shapenet, dim_remain=2)
                ax = fig.add_subplot(122)
                # ax.set_aspect('equal')
                ax.scatter(
                        sm_viz[:, 0],
                        sm_viz[:, 1],
                        c=labels_shapenet[:, 0],
                        alpha=0.4,
                        cmap='gist_rainbow')

                fig.savefig(output_path + '/z_feat_shapenet.png', transparent=True)
                plt.clf()

                t_i += 1

            if step_i % save_step == 0:

                # Save the variables to disk.
                # We should set global_step=batch_i if we want several ckpt
                saver.save(sess, output_path + "/" + ckpt_name,
                           global_step=None,
                           write_meta_graph=False)
                if softmax:
                    acc = sess.run(
                            ae['acc'],
                            feed_dict={
                                ae['x']: test_xs_img,
                                ae['label']: test_xs_label[:, 0],
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
                    summary = sess.run(
                            ae['merged'],
                            feed_dict={
                                    ae['x']: batch_xs_img,
                                    ae['t']: batch_xs_obj,
                                    ae['label']: batch_xs_label[:, 0],
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
