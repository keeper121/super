import skimage.transform
from skimage.io import imsave, imread
import os
from os import listdir, path
from os.path import isfile, join
import tensorflow as tf
import ntpath
import warnings
from matplotlib import pyplot as PLT
import numpy as np
#constants
__need_blurred_image__ = 1          # flag to generate blurred image
__upscale_coefficient__ = 2         # coefficient for image blurring
__tile_size__ = 64;

#open directory with images
def get_directory(folder):
    foundfile = []

    for path, subdirs, files in os.walk(folder):
        for name in files:
            found = os.path.join(path, name)
            if name.endswith('.jpg'):
                foundfile.append(found)
        break

    foundfile.sort()
    return foundfile

# perform 0.5x scale
# perform 2x scale
def get_blurred_image(_path, _coefficient):
    img = imread(_path)
    img = skimage.transform.rescale(img, 1.0/_coefficient)
    img = skimage.transform.rescale(img, _coefficient)
    return img

#do tiles
def crop_tiles_in_folder(_path, _image, _tile_size):
    i = 0
    for h in range(0, _image.shape[0], _tile_size):
        for w in range(0, _image.shape[1], _tile_size):
            w_end = w + _tile_size
            h_end = h + _tile_size
            imsave(_path + "/" + os.path.splitext(ntpath.basename(image))[0] + '_tiles_{0}.jpg'.format(i), _image[w:w_end, h:h_end])
            i += 1

#clear directory
def clear_directory(_path):
    for path, subdirs, files in os.walk(_path):
        for name in files:
            found = os.path.join(path, name)
            os.unlink(found)
        break

#--------------------------------------#

#original images
original_images = get_directory("images")

if __need_blurred_image__:
    clear_directory("out/blurred_tiles")
    clear_directory("out/original_tiles")
    for image in original_images:
        print (image)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blurred_image = get_blurred_image(image, __upscale_coefficient__)
            # resize to 256x256
            blurred_image = skimage.transform.resize(blurred_image, (256, 256))
            original_image = imread(image)
            original_image = skimage.transform.resize(original_image, (256, 256))
            # crop tiles
            crop_tiles_in_folder("out/blurred_tiles", blurred_image, __tile_size__)
            crop_tiles_in_folder("out/original_tiles", original_image, __tile_size__)

# blurred images tiles
blurred_images = get_directory("out/blurred_tiles")
blurred_images_tiles = []

for image in blurred_images:
    image = imread(image)
    blurred_images_tiles.append(image)

# original images tiles
original_images = get_directory("out/original_tiles")
original_images_tiles = []
for image in original_images:
    image = imread(image)
    original_images_tiles.append(image)
#--------------------------------------#

#--------------------------------------#
#tensor flow configuration

#parameters
f1 = 9
f2 = 1
f3 = 5
n1 = 64
n2 = 32
learning_rate = 0.0001
training_iters = 100000
batch_size = 5
display_step = 10
n_input = n_output = len(blurred_images)
# tf Graph input
x = tf.placeholder(tf.float32, [None, __tile_size__, __tile_size__, 3])
y = tf.placeholder(tf.float32, [None, __tile_size__, __tile_size__, 3])

def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

#create model

def model(_X, _weights, _biases):
    # Convolution Layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])

    # Convolution Layer
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])

    # Convolution Layer
    conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])
    return conv3

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([f1, f1, 3, n1])), # 1 input, n1 outputs
    'wc2': tf.Variable(tf.random_normal([f2, f2, n1, n2])), # n1 inputs, n2 outputs
    'wc3': tf.Variable(tf.random_normal([f3, f3, n2, 3])), # n2 inputs, 1 outputs
}

biases = {
    'bc1': tf.Variable(tf.random_normal([n1])),
    'bc2': tf.Variable(tf.random_normal([n2])),
    'bc3': tf.Variable(tf.random_normal([3])),
}

# Construct model
pred = model(x, weights, biases)
count_batch = n_input / batch_size

# get crop size

crop_size = ((((n1 - n2) / 2.0) - 3.0) / 2.0)
# crop_size = ((n1 - n2 - 3.0) / 2.0)

#slice_y = y[:, crop_size:__tile_size__ - crop_size, crop_size:__tile_size__ - crop_size, :]
#slice_pred = pred[:, crop_size:__tile_size__ - crop_size, crop_size:__tile_size__ - crop_size, :]


def cost_func(pred, y, crop_size):
    i = 0
    sum = tf.Variable(0.0)

    #get center
    while i < batch_size:
        slice_y = tf.image.central_crop(y[i,:,:,:], crop_size / __tile_size__)
        slice_pred = tf.image.central_crop(pred[i,:,:,:], crop_size / __tile_size__)
        tf.add(sum, tf.reduce_sum(tf.pow(tf.sub(slice_pred, slice_y), 2)))
        i = i + 1

    return tf.div(sum, (3 * __tile_size__ * __tile_size__ * count_batch * batch_size))


# Define loss and optimizer
# Euclidean distance
cost = cost_func(pred, y, crop_size)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(pred, y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
show_img = 1

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step < count_batch:
        batch_xs = blurred_images_tiles[batch_size * (step - 1):batch_size * step]
        batch_ys = original_images_tiles[batch_size * (step - 1):batch_size * step]

        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        # Calculate batch loss
        loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
        print "Iter " + str(step) + ", Training Accuracy = " + str(acc) + ", loss = " + str(loss)
        step += 1
    print "Optimization Finished!"

    # Test model
    print "Accuracy:", accuracy.eval({x: blurred_images_tiles[:count_batch * batch_size], y: original_images_tiles[:count_batch * batch_size]})
#--------------------------------------#



