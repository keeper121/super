# coding=utf-8
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pylab
import sp_util

# --------------------------------------#
# tensor flow
__tile_size__ = 33
_crop_size = 12  # for convolutional
_image_resize_height = 330
_image_resize_width = 330
_channel = 3  # RGB - 3 / grayscale - 1
# parameters
f1 = 9  # 1st convolutuonal kernel size
f2 = 1  # 2nd convolutuonal kernel size
f3 = 5  # 3rd convolutuonal kernel size
n1 = 64
n2 = 32
learning_rate = 0.000001
batch_size = 64
print_out = 10

wc1 = tf.Variable(tf.random_normal([f1, f1, _channel, n1], mean=0, stddev=0.05), name="wc1")  # 1 input, n1 outputs
wc2 = tf.Variable(tf.random_normal([f2, f2, n1, n2], mean=0, stddev=0.05), name="wc2")  # n1 inputs, n2 outputs
wc3 = tf.Variable(tf.random_normal([f3, f3, n2, _channel], mean=0, stddev=0.05), name="wc3")  # n2 inputs, 1 outputs

bc1 = tf.Variable(tf.random_normal(shape=[n1], mean=0.0, stddev=0.01), name="bc1")
bc2 = tf.Variable(tf.random_normal(shape=[n2], mean=0.0, stddev=0.01), name="bc2")
bc3 = tf.Variable(tf.random_normal(shape=[_channel], mean=0.0, stddev=0.01), name="bc3")

# tf Graph input
x = tf.placeholder(tf.float32, [None, __tile_size__, __tile_size__, _channel])
y = tf.placeholder(tf.float32, [None, __tile_size__ - _crop_size * 2, __tile_size__ - _crop_size * 2, _channel])


def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


# create model
def model(_X, wc1, wc2, wc3, bc1, bc2, bc3):
    # Convolution Layer
    conv1 = conv2d(_X, wc1, bc1)

    # Convolution Layer
    conv2 = conv2d(conv1, wc2, bc2)

    # Convolution Layer
    conv3 = conv2d(conv2, wc3, bc3)
    return conv3


def train(blurred_images_tiles, original_images_tiles, model_save_name):
    n_input = len(blurred_images_tiles)
    # Construct model
    count_batch = n_input / batch_size

    # Convolution Layer for show
    conv1 = conv2d(x, wc1, bc1)

    # Convolution Layer
    conv2 = conv2d(conv1, wc2, bc2)

    # Convolution Layer
    conv3 = conv2d(conv2, wc3, bc3)

    pred = model(x, wc1, wc2, wc3, bc1, bc2, bc3)
    slice_pred = pred[:, _crop_size:__tile_size__ - _crop_size, _crop_size:__tile_size__ - _crop_size, :]
    """
    # euclid distance - cost function
    sub = tf.abs(tf.sub(slice_pred, y))
    # sum over all dimensions except batch dim
    sum = tf.reduce_sum(sub, [1, 2, 3])
    # take mean over batch
    loss = tf.reduce_mean(tf.cast(sum, tf.float32))
    """

    reduce_sum = tf.reduce_sum(tf.abs(tf.square(tf.sub(slice_pred, y))))
    cost = tf.div(reduce_sum,
                  (_channel * (__tile_size__ - _crop_size * 2) * (__tile_size__ - _crop_size * 2) * batch_size))
    # Define loss and optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    accuracy = tf.reduce_mean(tf.cast(tf.sqrt(cost), tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()

    # Launch the graph

    # config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(init)
        writer = tf.train.SummaryWriter("/tmp/log", sess.graph)

        step = 1
        # Keep training until reach max iterations
        while step <= count_batch:
            # norming batches
            batch_xs_names = blurred_images_tiles[batch_size * (step - 1):batch_size * step]
            batch_ys_names = original_images_tiles[batch_size * (step - 1):batch_size * step]
            # get image batch
            batch_xs = sp_util.read_image(filenames=batch_xs_names, channels=_channel, need_crop=0,
                                          tile_size=__tile_size__, crop_size=_crop_size)
            batch_ys = sp_util.read_image(filenames=batch_ys_names, channels=_channel, need_crop=1,
                                          tile_size=__tile_size__, crop_size=_crop_size)

            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

            if step % print_out == 0:
                print "Iter " + str(step) + ", Training Accuracy = " + str(acc) + ", loss = " + str(loss)
                c1 = sess.run(conv1, feed_dict={x: batch_xs})
                c2 = sess.run(conv2, feed_dict={x: batch_xs})
                c3 = sess.run(conv3, feed_dict={x: batch_xs})
                print "1st conv avg:", np.average(c1), "; 2nd conv avg:", np.average(c2), "; 3rd conv avg:", np.average(
                    c3)
                print "bc3:", bc3.eval()
                print "complete:", float(step) / count_batch * 100.0, "%"
                print "_________________________________________________________"
                print

            step += 1

        print "Optimization Finished!"

        print "Last Biases:", bc3.eval()
        # Save model weights to disk
        save_path = saver.save(sess, model_save_name)
        print "Model saved in file: %s" % save_path

        writer.flush()
        writer.close()


# try get super resolutional from test image
def evaluate_model(model_name, image_path):
    # open blurred image
    blurred_image = sp_util.read_image(filenames=[image_path], channels=_channel, need_crop=0, tile_size=__tile_size__,
                                       crop_size=_crop_size)
    saver = tf.train.Saver()
    val = model(x, wc1, wc2, wc3, bc1, bc2, bc3)

    with tf.Session() as sess:
        saver.restore(sess, model_name)
        print("Model restored.")
        pbc3 = sess.run(bc3)
        print "Restored last Biases:", pbc3

        res = sess.run(val, feed_dict={x: blurred_image})
        print "output avg:", np.average(res)

        res = tf.squeeze(res).eval()
        blurred_image = tf.squeeze(blurred_image).eval()

        print np.average(res)

        ## TODO add grayscale draw
        pylab.title('Images')
        # blurred image
        f = pylab.figure()
        f.add_subplot(2, 1, 1)
        pylab.imshow(blurred_image)

        # restored image
        f.add_subplot(2, 1, 2)
        pylab.imshow(res)
        pylab.show()

# --------------------------------------#
need_train = 1
need_evaluate = 1

print "start"
# main code
if (need_train):
    blurred_tiles, original_tiles = sp_util.generate_dataset(_image_resize_height, _image_resize_width, __tile_size__)
    train(blurred_tiles, original_tiles, "model.ckpt")

# test
if (need_evaluate):
    evaluate_model("model.ckpt", "test.jpg")
