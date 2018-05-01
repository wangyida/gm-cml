# First check the Python version

# Now get necessary libraries
from libs.train_vae import train_vae

# Import Tensorflow
import tensorflow as tf

# This cell includes the provided libraries from the zip file
# and a library for displaying images from ipython, which
# we will use to display the gif


# Train it!  Change these parameters!
# n_code=72, here 72 = 12 * (8-2)
def test_shapenet():
    train_vae(files="./list_annotated_shapenet.csv",
                  input_shape=[116, 116, 3],
                  output_shape=[116, 116, 3],
                  learning_rate=0.0001,
                  batch_size=64,
                  n_epochs=50,
                  crop_shape=[112, 112],
                  crop_factor=1.0,
                  n_filters=[64, 64, 64, 128, 128],
                  n_hidden=128,
                  n_code=32,
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
                  ckpt_name="vae.ckpt",
                  output_path="results_gmcml")

if __name__ == '__main__':
    test_shapenet()
