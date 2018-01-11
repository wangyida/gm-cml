# First check the Python version

# Now get necessary libraries
from libs import vae

# Import Tensorflow
import tensorflow as tf

# This cell includes the provided libraries from the zip file
# and a library for displaying images from ipython, which
# we will use to display the gif


# Train it!  Change these parameters!
# n_code=72, here 72 = 12 * (8-2)
tf.reset_default_graph()
vae.train_vae(files="./list_annotated_shapenet.csv",
              input_shape=[116, 116, 3],
              output_shape=[116, 116, 3],
              learning_rate=0.0001,
              batch_size=36,
              n_epochs=50,
              n_examples=10,
              crop_shape=[112, 112],
              crop_factor=1,
              n_filters=[128, 128, 128, 128, 128],
              n_hidden=128,
              n_code=64,
              denoising=False,
              convolutional=True,
              variational=True,
              softmax=True,
              classifier='squeezenet',
              filter_sizes=[3, 3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=2500,
              save_step=500,
              ckpt_name="./vae.ckpt")
