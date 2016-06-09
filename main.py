import ntpath
import os
import warnings

import numpy as np
import skimage.transform
import tensorflow as tf
from skimage.io import imsave, imread
from PIL import Image
__tile_size__ = 32
_crop_size = 12 # for convolutional

class SuperResolutionDataSet:
    #constants
    __need_blurred_image__ = 1          # flag to generate blurred image
    __upscale_coefficient__ = 2         # coefficient for image blurring
    _original_image_directory = "images"
    _original_tiles_directory = "out/original_tiles"
    _blurred_tiles_directory = "out/blurred_tiles"



    #open directory with images
    def get_directory(self, folder):
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
    def get_blurred_image(self, _path, _coefficient):
        img = imread(_path)
        img = skimage.transform.rescale(img, 1.0/_coefficient)
        img = skimage.transform.rescale(img, _coefficient)
        return img

    #do tiles
    def crop_tiles_in_folder(self, _path, _image, _tile_size):
        i = 0
        for h in range(0, _image.shape[0], _tile_size):
            for w in range(0, _image.shape[1], _tile_size):
                w_end = w + _tile_size
                h_end = h + _tile_size
                imsave(_path + "/" + os.path.splitext(ntpath.basename(image))[0] + '_tiles_{0}.jpg'.format(i), _image[w:w_end, h:h_end])
                i += 1

    #clear directory
    def clear_directory(self, _path):
        for path, subdirs, files in os.walk(_path):
            for name in files:
                found = os.path.join(path, name)
                os.unlink(found)
            break

    #--------------------------------------#

    def centeredCrop(self, img, new_height, new_width):
        width =  np.size(img,1)
        height =  np.size(img,0)
        left = int(np.ceil((width - new_width)/2.))
        top = int(np.ceil((height - new_height)/2.))
        right = int(np.floor((width + new_width)/2.))
        bottom = int(np.floor((height + new_height)/2.))

        cImg = img[top:bottom, left:right]
        return cImg


    def generate_dataset(self):
        #original images
        original_images = self.get_directory(self._original_image_directory)
        blurred_images_tiles = []
        original_images_tiles = []
        if self.__need_blurred_image__:
            self.clear_directory(self._original_tiles_directory)
            self.clear_directory(self._blurred_tiles_directory)
            for image in original_images:
                print (image)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    blurred_image = self.get_blurred_image(image, self.__upscale_coefficient__)
                    # resize to 256x256
                    blurred_image = skimage.transform.resize(blurred_image, (256, 256))
                    original_image = imread(image)
                    original_image = skimage.transform.resize(original_image, (256, 256))
                    # crop tiles
                    self.crop_tiles_in_folder(self._blurred_tiles_directory, blurred_image, __tile_size__)
                    self.crop_tiles_in_folder(self._original_tiles_directory, original_image, __tile_size__)

                    # blurred images tiles
                    blurred_images = self.get_directory(self._blurred_tiles_directory)


                    for image in blurred_images:
                        image = imread(image)
                        blurred_images_tiles.append(image)
                    # original images tiles
                    original_images = self.get_directory("out/original_tiles")

                    for image in original_images:
                        image = imread(image)
                        original_images_tiles.append(self.centeredCrop(image, __tile_size__ - _crop_size * 2, __tile_size__ - _crop_size * 2))

        return blurred_images_tiles, original_images_tiles


#--------------------------------------#

class SuperResolutionModel:
    #--------------------------------------#
    #tensor flow
    #parameters
    f1 = 9                              # 1st convolutuonal kernel size
    f2 = 1                              # 2nd convolutuonal kernel size
    f3 = 5                              # 3rd convolutuonal kernel size
    n1 = 64
    n2 = 32
    learning_rate = 0.001
    #training_iters = 100000
    batch_size = 8
    #display_step = 10
    #n_input = n_output = len(blurred_images)

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

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, __tile_size__, __tile_size__, 3])
    y = tf.placeholder(tf.float32, [None, __tile_size__ - _crop_size * 2, __tile_size__ - _crop_size * 2, 3])

    def conv2d(self, img, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

    #create model

    def model(self,_X, _weights, _biases):
        # Convolution Layer
        conv1 = self.conv2d(_X, _weights['wc1'], _biases['bc1'])

        # Convolution Layer
        conv2 = self.conv2d(conv1, _weights['wc2'], _biases['bc2'])

        # Convolution Layer
        conv3 = self.conv2d(conv2, _weights['wc3'], _biases['bc3'])
        return conv3

    # euclid distance
    def cost_func(self, pred, y, crop_size, count_batch):
        slice_pred = pred[:, crop_size:__tile_size__ - crop_size, crop_size:__tile_size__ - crop_size, :]
        return tf.div(tf.reduce_sum(tf.pow(tf.sub(slice_pred, y), 2)), (3 * (__tile_size__ - crop_size) * (__tile_size__ - crop_size) * count_batch * self.batch_size))


    def train(self, blurred_images_tiles, original_images_tiles, model_save_name):
        n_input = len(blurred_images_tiles)
        # Construct model
        pred = self.model(self.x, self.weights, self.biases)
        count_batch = n_input / self.batch_size
        # Define loss and optimizer

        # Euclidean distance
        cost = self.cost_func(pred, self.y, _crop_size, count_batch)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)

        euclid = self.cost_func(pred, self.y, _crop_size, count_batch)
        accuracy = tf.reduce_mean(tf.cast(tf.sqrt(euclid), tf.float32))

        # Initializing the variables
        init = tf.initialize_all_variables()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step <= count_batch:
                batch_xs = blurred_images_tiles[self.batch_size * (step - 1):self.batch_size * step]
                batch_ys = original_images_tiles[self.batch_size * (step - 1):self.batch_size * step]

                # Fit training using batch data
                sess.run(optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys})
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={self.x: batch_xs, self.y: batch_ys})
                print "Iter " + str(step) + ", Training Accuracy = " + str(acc) + ", loss = " + str(loss)
                step += 1
                print "Optimization Finished!"

                # Test model
                print "Accuracy:", accuracy.eval({self.x: blurred_images_tiles[:count_batch * self.batch_size], self.y: original_images_tiles[:count_batch * self.batch_size]}) / count_batch

                # Save model weights to disk
                save_path = saver.save(sess, model_save_name)
                print "Model saved in file: %s" % save_path

    # try get super resolutional from test image
    def evaluate_model(self, model_name, image_path):
        #open blurred image
        blurred_image = [imread(image_path)]
        saver = tf.train.Saver()
        val = self.model(self.x, self.weights, self.biases)
        #y = tf.Variable(tf.random_normal([1, __tile_size__, __tile_size__, 3]))
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            saver.restore(sess, model_name)
            print("Model restored.")
            sess.run(init)
            res = sess.run(val, feed_dict={self.x:blurred_image})
            img = Image.fromarray(res[0], 'RGB')
            img.show()
#--------------------------------------#

# main code
blurred_tiles, original_tiles = SuperResolutionDataSet().generate_dataset()
SuperResolutionModel().train(blurred_tiles, original_tiles, "model.ckpt")

#test
SuperResolutionModel().evaluate_model("model.ckpt")