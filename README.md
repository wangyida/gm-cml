# Adaptive Noise Assisted Conjugating Generative Model
TensorFlow implementation for Adaptive noise assisted Conjugating Generative Model based on Conditional Variational Autoencoder, this project is implemented based on VAE code pool of Parag K. Mital from Kadenze course on Tensorflow and modified by Yida Wang for the paper of 'Conjugating Generative Model for Object Recognition Based on 3D Models'.

Copyright reserved for Yida Wang from BUPT.

## Figures in the paper
### Method Pipeline

This is the basic pipeline for AD-CGM

![Pipeline](https://github.com/wangyida/pynote/blob/master/vae/images/pipeline_tip.jpg)

### Target

There are two set of rendered images from 3D models, here we use *VTK* for rendering.

![](https://github.com/wangyida/pynote/blob/master/vae/images/target.jpg)

### Results on ShapeNet

Those images are input of rendered images from ShapeNet and the reconstruction result of synthetic images.

![Reconstruction](https://github.com/wangyida/pynote/blob/master/vae/images/shapenet_recon.jpg)

final distribution of AD-CGM from classification sub-network visualized using PCA

![ShapeNet](https://github.com/wangyida/pynote/blob/master/vae/images/lowdim_ADCGM_shapenet.jpg)

### Results on ImageNet

The reconstruction result of real images from ImageNet.

![Reconstruction](https://github.com/wangyida/pynote/blob/master/vae/images/imagenet_recon.jpg)

final distribution of AD-CGM from classification sub-network visualized using PCA.

![ImageNet](https://github.com/wangyida/pynote/blob/master/vae/images/lowdim_ADCGM_imagenet.jpg)


## Author info

### Basic info

Yida Wang

Email: yidawang.cn@gmail.com

### Publications

[ZigzagNet: Efficient Deep Learning for Real Object Recognition Based on 3D Models](https://www.researchgate.net/profile/Yida_Wang/publications?sorting=recentlyAdded)

[Self-restraint Object Recognition by Model Based CNN Learning](http://ieeexplore.ieee.org/document/7532438/)

[Face Recognition Using Local PCA Filters](http://link.springer.com/chapter/10.1007%2F978-3-319-25417-3_5)

[CNTK on Mac: 2D Object Restoration and Recognition Based on 3D Model](https://www.microsoft.com/en-us/research/academic-program/microsoft-open-source-challenge/)

[Large-Scale 3D Shape Retrieval from ShapeNet Core55](https://shapenet.cs.stanford.edu/shrec16/shrec16shapenet.pdf)

### Extracurriculars

Contributor for OpenCV and [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn);

Google Summer of Codes successor from 2015 to 2016

### Personal Links

[ResearchGate](https://www.researchgate.net/profile/Yida_Wang), [Github](https://github.com/wangyida), [GSoC 2016](https://summerofcode.withgoogle.com/archive/2016/projects/4623962327744512/), [GSoC 2015](https://www.google-melange.com/archive/gsoc/2015/orgs/opencv/projects/wangyida.html)

***

## Additional Libs
There is also implementation of ZigzagNet and SqueezeNet for compact deep learning for classification.

## Codes Explanation

### Training and Testing Strategies
network input / placeholders for train (bn) and dropout
```python
    x_img = tf.placeholder(tf.float32, input_shape, 'x_img')
    x_obj = tf.placeholder(tf.float32, input_shape, 'x_obj')
```

Input of the reconstruction network
```python
    current_input1 = utils.corrupt(x_img)*corrupt_rec + x_img*(1-corrupt_rec) \
                if (denoising and phase_train is not None) else x_img
    current_input1.set_shape(x_img.get_shape())
    # 2d -> 4d if convolution
    current_input1 = utils.to_tensor(current_input1) \
                if convolutional else current_input1
```

Encoder
```python
    for layer_i, n_output in enumerate(n_filters):
        with tf.variable_scope('encoder/{}'.format(layer_i)):
            shapes.append(current_input1.get_shape().as_list())
            if convolutional:

    with tf.variable_scope('variational'):
        if variational:
```

Decoder
```python
    for layer_i, n_output in enumerate(n_filters[1:]):
        with tf.variable_scope('decoder/{}'.format(layer_i)):
            shape = shapes[layer_i + 1]
            if convolutional:
```

Loss finctions of VAE and softmax
```python
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

    # Alexnet for clasification based on softmax using TensorFlow slim
    if softmax:
```

There are several optional choices for classification network. Just modify the parameter ```classifier```.
```python

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
```

Here we must set corrupt_rec and corrupt_cls as 0 to find a proper ratio of variance to feed for variable var_prob. We use tanh as non-linear function for ratio of Vars from the reconstructed channels and original channels.
```python
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
```

Main API for training and testing. General purpose training of a (Variational) (Convolutional) Autoencoder.
```python
def train_vae(files_img, files_obj, input_shape):
    """
    Parameters
    ----------
    files : list of strings
        List of paths to images.
    input_shape : list
        Must define what the input image's shape is.
    use_csv = bool, optional
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
```

### Visualization for Outputs and Parameters

We can visualize filters, reconstruction channels and also outputs according to latent variables.

Plot example reconstructions
```python
recon = sess.run(
    ae['y'], feed_dict={
                ae['x_img']: test_xs_img,
                ae['train']: False,
                ae['keep_prob']: 1.0,
                ae['corrupt_rec']: 0,
                ae['corrupt_cls']: 0})
utils.montage(recon.reshape([-1] + crop_shape),
              'recon_%08d.png' % t_i)
```

Plot filters
```python
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
```            
