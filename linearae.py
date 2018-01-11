# First check the Python version
import sys
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n' \
          'You should consider updating to Python 3.4.0 or ' \
          'higher as the libraries built for this course ' \
          'have only been tested in Python 3.4 and higher.\n')
    print('Try installing the Python 3.5 version of anaconda '
          'and then restart `jupyter notebook`:\n' \
          'https://www.continuum.io/downloads\n\n')

# Now get necessary libraries
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    import IPython.display as ipyd
except ImportError:
    print('You are missing some packages! ' \
          'We will try installing them before continuing!')
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    import IPython.display as ipyd
    print('Done!')

# Import Tensorflow
try:
    import tensorflow as tf
except ImportError:
    print("You do not have tensorflow installed!")
    print("Follow the instructions on the following link")
    print("to install tensorflow before continuing:")
    print("")
    print("https://github.com/pkmital/CADL#installation-preliminaries")

# This cell includes the provided libraries from the zip file
# and a library for displaying images from ipython, which
# we will use to display the gif
try:
    from libs import utils, gif, datasets, dataset_utils, vae, dft
except ImportError:
    print("Make sure you have started notebook in the same directory" +
          " as the provided zip file which includes the 'libs' folder" +
          " and the file 'utils.py' inside of it.  You will NOT be able"
          " to complete this assignment unless you restart jupyter"
          " notebook inside the directory created by extracting"
          " the zip file or cloning the github repo.")

# We'll tell matplotlib to inline any drawn figures like so:

# See how this works w/ Celeb Images or try your own dataset instead:

class ScanFile(object):
    def __init__(self,directory,prefix=None,postfix='.jpg'):
        self.directory=directory
        self.prefix=prefix
        self.postfix=postfix

    def scan_files(self):
        files_list=[]

        for dirpath,dirnames,filenames in os.walk(self.directory):
            '''''
            dirpath is a string, the path to the directory.
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..').
            filenames is a list of the names of the non-directory files in dirpath.
            '''
            for special_file in filenames:
                if self.postfix:
                    special_file.endswith(self.postfix)
                    files_list.append(os.path.join(dirpath,special_file))
                elif self.prefix:
                    special_file.startswith(self.prefix)
                    files_list.append(os.path.join(dirpath,special_file))
                else:
                    files_list.append(os.path.join(dirpath,special_file))

        return files_list

    def scan_subdir(self):
        subdir_list=[]
        for dirpath,dirnames,files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list

def get_myown_files(direc):
    files = [os.path.join(direc, file_i)
             for file_i in os.listdir(direc)
             if '.jpg' in file_i]
    return files

def get_myown_imgs(direc):
    scan=ScanFile(direc)
    files_img=scan.scan_files()
    return [plt.imread(f_i) for f_i in files_img]

# Write a function to preprocess/normalize an image, given its dataset object
# (which stores the mean and standard deviation!)
def preprocess(img, ds):
    norm_img = (img - ds.mean()) / ds.std()
    return norm_img

# Write a function to undo the normalization of an image, given its dataset object
# (which stores the mean and standard deviation!)
def deprocess(norm_img, ds):
    img = norm_img * ds.std() + ds.mean()
    return img

direc = '/home/yida/Documents/buildboat/slic_superpixel/data/annotated_img/aeroplane'
myown_img = get_myown_imgs(direc)
direc = '/home/yida/Documents/buildboat/slic_superpixel/data/annotated_obj/aeroplane'
myown_obj = get_myown_imgs(direc)


# Then resize the square image to 100 x 100 pixels
myown_img = [resize(img_i, (100, 100, 3)) for img_i in myown_img]
myown_obj = [resize(img_i, (100, 100, 3)) for img_i in myown_obj]
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(myown_img))

# Then convert the list of images to a 4d array (e.g. use np.array to convert a list to a 4d array):
Xs = np.array(myown_img).copy()*255

print(Xs.shape)
assert(Xs.ndim == 4 and Xs.shape[1] <= 100 and Xs.shape[2] <= 100)

ds_img = datasets.Dataset(Xs)

# Then convert the list of images to a 4d array (e.g. use np.array to convert a list to a 4d array):
Xs = np.array(myown_obj).copy()*255

print(Xs.shape)
assert(Xs.ndim == 4 and Xs.shape[1] <= 100 and Xs.shape[2] <= 100)

ds_obj = datasets.Dataset(Xs)

for (X_img, y) in ds_img.train.next_batch(batch_size=25):
    print(X_img.shape)
for (X_obj, y) in ds_obj.train.next_batch(batch_size=25):
    print(X_obj.shape)

# Just to make sure that you've coded the previous two functions correctly:
assert(np.allclose(deprocess(preprocess(ds_img.X[0], ds_img), ds_img), ds_img.X[0]))

# Calculate the number of features in your image.
# This is the total number of pixels, or (height x width x channels).
n_features = ds_img.X[0].shape[0]*ds_img.X[0].shape[1]*ds_img.X[0].shape[2]
print(n_features)

encoder_dimensions = [2048, 512, 128, 2]

tf.reset_default_graph()
X_img = tf.placeholder(tf.float32, shape=(None, n_features), name="X_img")
X_obj = tf.placeholder(tf.float32, shape=(None, n_features), name="X_obj")
assert(X_img.get_shape().as_list() == [None, n_features])
assert(X_obj.get_shape().as_list() == [None, n_features])

def encode(X, dimensions, activation=tf.nn.tanh):
    # We're going to keep every matrix we create so let's create a list to hold them all
    Ws = []

    # We'll create a for loop to create each layer:
    for layer_i, n_output in enumerate(dimensions):

        # TODO: just like in the last session,
        # we'll use a variable scope to help encapsulate our variables
        # This will simply prefix all the variables made in this scope
        # with the name we give it.  Make sure it is a unique name
        # for each layer, e.g., 'encoder/layer1', 'encoder/layer2', or
        # 'encoder/1', 'encoder/2',...
        with tf.variable_scope("encoder/layer/{}".format(layer_i)):

            # TODO: Create a weight matrix which will increasingly reduce
            # down the amount of information in the input by performing
            # a matrix multiplication.  You can use the utils.linear function.
            h, W = utils.linear(X, n_output, activation = activation)

            # Finally we'll store the weight matrix.
            # We need to keep track of all
            # the weight matrices we've used in our encoder
            # so that we can build the decoder using the
            # same weight matrices.
            Ws.append(W)

            # Replace X with the current layer's output, so we can
            # use it in the next layer.
            X = h

    z = X
    return Ws, z

# Then call the function
Ws, z = encode(X_img, encoder_dimensions)

# And just some checks to make sure you've done it right.
assert(z.get_shape().as_list() == [None, 2])
assert(len(Ws) == len(encoder_dimensions))

[op.name for op in tf.get_default_graph().get_operations()]

[W_i.get_shape().as_list() for W_i in Ws]

z.get_shape().as_list()

# We'll first reverse the order of our weight matrices
decoder_Ws = Ws[::-1]

# then reverse the order of our dimensions
# appending the last layers number of inputs.
decoder_dimensions = encoder_dimensions[::-1][1:] + [n_features]
print(decoder_dimensions)

assert(decoder_dimensions[-1] == n_features)

def decode(z, dimensions, Ws, activation=tf.nn.tanh):
    current_input = z
    for layer_i, n_output in enumerate(dimensions):
        # we'll use a variable scope again to help encapsulate our variables
        # This will simply prefix all the variables made in this scope
        # with the name we give it.
        with tf.variable_scope("decoder/layer/{}".format(layer_i)):

            # Now we'll grab the weight matrix we created before and transpose it
            # So a 3072 x 784 matrix would become 784 x 3072
            # or a 256 x 64 matrix, would become 64 x 256
            W = tf.transpose(Ws[layer_i])

            # Now we'll multiply our input by our transposed W matrix
            h = tf.matmul(current_input, W)

            # And then use a relu activation function on its output
            current_input = activation(h)

            # We'll also replace n_input with the current n_output, so that on the
            # next iteration, our new number inputs will be correct.
            n_input = n_output
    Y = current_input
    return Y

Y = decode(z, decoder_dimensions, decoder_Ws)

[op.name for op in tf.get_default_graph().get_operations()
 if op.name.startswith('decoder')]

Y.get_shape().as_list()

# Calculate some measure of loss, e.g. the pixel to pixel absolute difference or squared difference
loss = tf.squared_difference(X_obj, Y)

# Now sum over every pixel and then calculate the mean over the batch dimension (just like session 2!)
# hint, use tf.reduce_mean and tf.reduce_sum
cost = tf.reduce_mean(tf.reduce_sum(loss, 1))

learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# (TODO) Create a tensorflow session and initialize all of our weights:
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Some parameters for training
batch_size = 50
n_epochs = 201
step = 40

# We'll try to reconstruct the same first 250 images and show how
# The network does over the course of training.
examples = ds_img.X[:225]

# We have to preprocess the images before feeding them to the network.
# I'll do this once here, so we don't have to do it every iteration.
test_examples = preprocess(examples, ds_img).reshape(-1, n_features)

# If we want to just visualize them, we can create a montage.
test_images = utils.montage(examples).astype(np.uint8)

# Store images so we can make a gif
gifs = []

# Now for our training:
for epoch_i in range(n_epochs):

    # Keep track of the cost
    this_cost = 0

    # Iterate over the entire dataset in batches
    for batch_X, _ in ds_img.train.next_batch(batch_size=batch_size):

        # (TODO) Preprocess and reshape our current batch, batch_X:
        this_batch = preprocess(batch_X, ds_img).reshape(-1, n_features)

        # Compute the cost, and run the optimizer.
        this_cost += sess.run([cost, optimizer], feed_dict={X_img: this_batch})[0]

    # Average cost of this epoch
    avg_cost = this_cost / ds_img.X.shape[0] / batch_size
    print(epoch_i, avg_cost)

    # Let's also try to see how the network currently reconstructs the input.
    # We'll draw the reconstruction every `step` iterations.
    if epoch_i % step == 0:

        # (TODO) Ask for the output of the network, Y, and give it our test examples
        recon = sess.run(Y, feed_dict={X_img: test_examples})

        # Resize the 2d to the 4d representation:
        rsz = recon.reshape(examples.shape)

        # We have to unprocess the image now, removing the normalization
        unnorm_img = deprocess(rsz, ds_img)

        # Clip to avoid saturation
        clipped = np.clip(unnorm_img, 0, 255)

        # And we can create a montage of the reconstruction
        recon = utils.montage(clipped).astype(np.uint8)

        # Store for gif
        gifs.append(recon)

        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(test_images)
        axs[0].set_title('Original')
        axs[1].imshow(recon)
        axs[1].set_title('Synthesis')
        fig.canvas.draw()
        plt.show()
