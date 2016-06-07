import skimage.transform
from skimage.io import imsave, imread
import os
from os import listdir, path
from os.path import isfile, join
from skimage.io import imread
import scipy
import tensorflow as tf
from scipy.misc import toimage

#parameters

f1 = 9
f2 = 1
f3 = 5
n1 = 64
n2 = 32
image_path = "test.jpg"


# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([f1, f1, 3, n1]), name="wc1"), # 1 input, n1 outputs
    'wc2': tf.Variable(tf.random_normal([f2, f2, n1, n2]), name="wc2"), # n1 inputs, n2 outputs
    'wc3': tf.Variable(tf.random_normal([f3, f3, n2, 3]), name="wc3"), # n2 inputs, 1 outputs
}

biases = {
    'bc1': tf.Variable(tf.random_normal([n1]), name="bc1"),
    'bc2': tf.Variable(tf.random_normal([n2]), name="bc2"),
    'bc3': tf.Variable(tf.random_normal([3]), name="bc3"),
}


#open blurred image
blurred_image = imread(image_path)


saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess, "model.ckpt")
  print("Model restored.")

  # 1 - convolutional
  output = scipy.ndimage.convolve(blurred_image, weights['wc1'].eval())

  # 2 - convolutional
  output = scipy.ndimage.convolve(output, weights['wc2'].eval())

  # 3 - convolutional
  output = scipy.ndimage.convolve(output, weights['wc3'].eval())

  toimage(output).show()