from skimage.io import imread
import tensorflow as tf
from PIL import Image

#parameters

f1 = 9
f2 = 1
f3 = 5
n1 = 64
n2 = 32
image_path = "test.jpg"
__tile_size__ = 64
crop_size = 12

# tf Graph input
x = tf.placeholder(tf.float32, [None, __tile_size__, __tile_size__, 3])


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
blurred_image = [imread(image_path)]
saver = tf.train.Saver()
val = model(x, weights, biases)
#y = tf.Variable(tf.random_normal([1, __tile_size__, __tile_size__, 3]))
init = tf.initialize_all_variables()

with tf.Session() as sess:
    saver.restore(sess, "model.ckpt")
    print("Model restored.")
    sess.run(init)
    res = sess.run(val, feed_dict={x:blurred_image})
    img = Image.fromarray(res[0], 'RGB')
    img.show()
