# coding=utf-8
import ntpath
import os
import warnings
import pylab
import numpy as np
import skimage
import skimage.transform
import tensorflow as tf
from skimage.io import imsave, imread
from PIL import Image
__tile_size__ = 64 #42
_crop_size = 12 # for convolutional

class SuperResolutionDataSet:
    #constants
    __need_blurred_image__ = 1          # flag to generate blurred image
    __upscale_coefficient__ = 3         # coefficient for image blurring
    _original_image_directory = "images"
    _original_tiles_directory = "out/original_tiles"
    _blurred_tiles_directory = "out/blurred_tiles"

    _image_resize_width = 256 #336
    _image_resize_height = 256

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
    def crop_tiles_in_folder_tf(self, _path, _image, height, width, image_name, _tile_size):
        i = 0
        for h in range(0, height, _tile_size):
            for w in range(0, width, _tile_size):
#                imsave(_path + "/" + os.path.splitext(ntpath.basename(image_name))[0] + '_tiles_{0}.jpg'.format(i), _image[w:w_end, h:h_end])
                image = tf.image.crop_to_bounding_box(_image, h, w, _tile_size, _tile_size);

                with tf.Session() as sess:
                    image = image.eval()
                    imsave(_path + "/" + os.path.splitext(ntpath.basename(image_name))[0] + '_tiles_{0}.jpg'.format(i), image)
                i += 1

    #do tiles
    def crop_tiles_in_folder(self, _path, _image, image_name, _tile_size):
        i = 0
        for h in range(0, _image.shape[0], _tile_size):
            for w in range(0, _image.shape[1], _tile_size):
                w_end = w + _tile_size
                h_end = h + _tile_size
                imsave(_path + "/" + os.path.splitext(ntpath.basename(image_name))[0] + '_tiles_{0}.jpg'.format(i), _image[w:w_end, h:h_end])
                i += 1

    #clear directory
    def clear_directory(self, _path):
        for path, subdirs, files in os.walk(_path):
            for name in files:
                found = os.path.join(path, name)
                os.unlink(found)
            break

    #--------------------------------------#

    def crop_or_pad(self, _image, height, width):
        img_h = _image.shape[0]
        img_w = _image.shape[1]

        pad_img = np.zeros(shape=(height, width, 3), dtype=_image.dtype)
        pad_h = abs(height - img_h) / 2.0
        pad_w = abs(width - img_w) / 2.0
        pad_top = int(pad_h) + 1 if (pad_h - int(pad_h) != 0) else pad_h
        pad_right = int(pad_w) + 1 if (pad_w - int(pad_w) != 0) else int(pad_w)

        if (img_h < height and img_w < width):
            pad_img[pad_h: -pad_top, pad_w: -pad_right, :] =_image

        if (img_h > height and img_w > width):
            pad_img = self.centeredCrop(_image, height, width)

        if (img_h < height and img_w > width):
            crimg = self.centeredCrop(_image, height, width)
            pad_img[pad_h: -pad_top, :, :] = crimg

        if (img_h > height and img_w < width):
            crimg = self.centeredCrop(_image, height, width)
            pad_img[:, pad_w: -pad_right, :] = crimg

        return pad_img

    def centeredCrop(self, _image, height, width):
        img_h = _image.shape[0]
        img_w = _image.shape[1]

        top = 0
        bottom = img_h
        left = 0
        right = img_w
        if img_h > height:
            top = int(np.floor((img_h - height)/2.))
            bottom = int(np.floor((img_h + height)/2.))

        if img_w > width:
            left = int(np.floor((img_w - width)/2.))
            right = int(np.floor((img_w + width)/2.))

        return _image[top:bottom, left:right]

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
                    image_name = image
                    original_image = imread(image)
                    blurred_image = self.get_blurred_image(image, self.__upscale_coefficient__)
                    shape = original_image.shape

                    blurred_image = self.crop_or_pad(blurred_image, self._image_resize_height, self._image_resize_width)
                    original_image = self.crop_or_pad(original_image, self._image_resize_height, self._image_resize_width)

                     # crop tiles
                    self.crop_tiles_in_folder(self._blurred_tiles_directory, blurred_image, image_name, __tile_size__)
                    self.crop_tiles_in_folder(self._original_tiles_directory, original_image, image_name, __tile_size__)
                    """
                    # resize
                    blurred_image = tf.image.resize_image_with_crop_or_pad(blurred_image, self._image_resize_height, self._image_resize_width)
                    original_image = tf.image.resize_image_with_crop_or_pad(original_image, self._image_resize_height, self._image_resize_width)

                    # crop tiles
                    self.crop_tiles_in_folder_tf(self._blurred_tiles_directory, blurred_image, self._image_resize_height, self._image_resize_width, image_name, __tile_size__)
                    self.crop_tiles_in_folder_tf(self._original_tiles_directory, original_image, self._image_resize_height, self._image_resize_width, image_name, __tile_size__)
                    """
                    print shape, "->", original_image.shape
            # blurred images tiles
            blurred_images = self.get_directory(self._blurred_tiles_directory)
            print str(len(blurred_images)) + "x2", "tiles saved"

            for image in blurred_images:
                image = imread(image)
                blurred_images_tiles.append(image)
            print "blurred tiles opened"
            # original images tiles
            original_images = self.get_directory(self._original_tiles_directory)
            for image in original_images:
                image = imread(image)
                original_images_tiles.append(self.centeredCrop(image, __tile_size__ - _crop_size * 2, __tile_size__ - _crop_size * 2))
            print "original tiles opened"

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
    learning_rate = 0.01
    batch_size = 8
    #n_input = n_output = len(blurred_images)
    # Store layers weight & bias

    wc1 = tf.Variable(tf.random_normal([f1, f1, 3, n1]), name="wc1") # 1 input, n1 outputs
    wc2 = tf.Variable(tf.random_normal([f2, f2, n1, n2]), name="wc2") # n1 inputs, n2 outputs
    wc3 = tf.Variable(tf.random_normal([f3, f3, n2, 3]), name="wc3") # n2 inputs, 1 outputs

    bc1 = tf.Variable(tf.random_normal([n1]), name="bc1")
    bc2 = tf.Variable(tf.random_normal([n2]), name="bc2")
    bc3 = tf.Variable(tf.random_normal([3]), name="bc3")

    test = tf.Variable(0.0, name="test")

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, __tile_size__, __tile_size__, 3])
    y = tf.placeholder(tf.float32, [None, __tile_size__ - _crop_size * 2, __tile_size__ - _crop_size * 2, 3])

    def conv2d(self, img, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

    #create model

    def model(self,_X, wc1, wc2, wc3, bc1, bc2, bc3):
        # Convolution Layer
        conv1 = self.conv2d(_X, wc1, bc1)

        # Convolution Layer
        conv2 = self.conv2d(conv1, wc2, bc2)

        # Convolution Layer
        conv3 = self.conv2d(conv2, wc3, bc3)
        return conv3

    # euclid distance
    def cost_func(self, pred, y, crop_size, count_batch):
        slice_pred = pred[:, crop_size:__tile_size__ - crop_size, crop_size:__tile_size__ - crop_size, :]
        return tf.div(tf.reduce_sum(tf.pow(tf.sub(slice_pred, y), 2)), (3 * (__tile_size__ - crop_size) * (__tile_size__ - crop_size) * self.batch_size))


    def train(self, blurred_images_tiles, original_images_tiles, model_save_name):
        n_input = len(blurred_images_tiles)
        # Construct model
        count_batch = n_input / self.batch_size
        with tf.name_scope('model'):
            pred = self.model(self.x, self.wc1, self.wc2, self.wc3, self.bc1, self.bc2, self.bc3)
            # Define loss and optimizer

            # Euclidean distance
            with tf.name_scope('cost'):
                cost = self.cost_func(pred, self.y, _crop_size, count_batch)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)

            with tf.name_scope('accuracy'):
                euclid = self.cost_func(pred, self.y, _crop_size, count_batch)
                accuracy = tf.reduce_mean(tf.cast(tf.sqrt(euclid), tf.float32))

        # Initializing the variables
        init = tf.initialize_all_variables()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            writer = tf.train.SummaryWriter("/tmp/log", sess.graph)

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

                model = sess.run(pred, feed_dict={self.x: batch_xs})
                print model
                print "Iter " + str(step) + ", Training Accuracy = " + str(acc) + ", loss = " + str(loss)
                step += 1

            print "Optimization Finished!"

            # Test model
            print "Accuracy:", accuracy.eval({self.x: blurred_images_tiles[:self.batch_size], self.y: original_images_tiles[:self.batch_size]})

            print "Last Biases:", self.bc3.eval()
            # Save model weights to disk
            save_path = saver.save(sess, model_save_name)
            print "Model saved in file: %s" % save_path

            writer.flush()
            writer.close()


    # try get super resolutional from test image
    def evaluate_model(self, model_name, image_path):
        #open blurred image
        blurred_image = skimage.transform.resize(imread(image_path), (__tile_size__, __tile_size__));
        blurred_image = [blurred_image]
        saver = tf.train.Saver()
        val = self.model(self.x, self.wc1, self.wc2, self.wc3, self.bc1, self.bc2, self.bc3)
        #y = tf.Variable(tf.random_normal([1, __tile_size__, __tile_size__, 3]))

        with tf.Session() as sess:
            saver.restore(sess, model_name)
            print("Model restored.")
            bc3 = sess.run(self.bc3)
            print "Restored last Biases:", bc3

            res = sess.run(val, feed_dict={self.x: blurred_image})

            print res
            img = Image.fromarray(res[0], 'RGB')
            f = pylab.figure()
            #blurred image
            f.add_subplot(2, 1, 1)
            image = Image.open(image_path)
            arr = np.asarray(image)
            pylab.imshow(arr)

            #restored image
            f.add_subplot(2, 1, 2)
            image = img
            arr = np.asarray(image)
            pylab.imshow(arr)


            pylab.title('Images')
            pylab.show()
#--------------------------------------#
need_train = 1
need_evaluate = 1

# main code
if (need_train):
    blurred_tiles, original_tiles = SuperResolutionDataSet().generate_dataset()
    SuperResolutionModel().train(blurred_tiles, original_tiles, "model.ckpt")

#test
if (need_evaluate):
    SuperResolutionModel().evaluate_model("model.ckpt", "test.jpg")