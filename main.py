# coding=utf-8
import ntpath
import os
import warnings
import tensorflow as tf
import numpy as np
import skimage
import skimage.transform
from matplotlib import pylab
from skimage.io import imsave, imread
from PIL import Image
__tile_size__ = 42
_crop_size = 12 # for convolutional

class SuperResolutionDataSet:
    #constants
    __need_blurred_image__ = 0          # flag to generate blurred image
    __upscale_coefficient__ = 2         # coefficient for image blurring
    _original_image_directory = "images"
    _original_tiles_directory = "out/original_tiles"
    _blurred_tiles_directory = "out/blurred_tiles"

    _image_resize_width = 336
    _image_resize_height = 336

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
            print "clear original tiles folder"
            self.clear_directory(self._original_tiles_directory)
            print "clear blurred tiles folder"
            self.clear_directory(self._blurred_tiles_directory)
            for image in original_images:
                print (image)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    image_name = image
                    original_image = imread(image)
                    blurred_image = self.get_blurred_image(image, self.__upscale_coefficient__)
                    shape = original_image.shape
                    #for cpu

                    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                        blurred_image = self.crop_or_pad(blurred_image, self._image_resize_height, self._image_resize_width)
                        original_image = self.crop_or_pad(original_image, self._image_resize_height, self._image_resize_width)
                        # crop tiles
                        self.crop_tiles_in_folder(self._blurred_tiles_directory, blurred_image, image_name, __tile_size__)
                        self.crop_tiles_in_folder(self._original_tiles_directory, original_image, image_name, __tile_size__)
                        print shape, "->", original_image.shape

                    """
                    #TODO make faster variant -> tf.while_loop
                    # for gpu
                    blurred_image = tf.image.resize_image_with_crop_or_pad(blurred_image, self._image_resize_height, self._image_resize_width)
                    original_image = tf.image.resize_image_with_crop_or_pad(original_image, self._image_resize_height, self._image_resize_width)

                    # crop tiles
                    self.crop_tiles_in_folder_tf(self._blurred_tiles_directory, blurred_image, self._image_resize_height, self._image_resize_width, image_name, __tile_size__)
                    self.crop_tiles_in_folder_tf(self._original_tiles_directory, original_image, self._image_resize_height, self._image_resize_width, image_name, __tile_size__)
                    print shape, "->", original_image.get_shape()
                    """
        # blurred images tiles
        blurred_images_tiles = self.get_directory(self._blurred_tiles_directory)
        original_images_tiles= self.get_directory(self._original_tiles_directory)
        print str(len(blurred_images_tiles)) + "x2", "tiles saved"
        """
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
        """
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
    learning_rate = 0.0001
    batch_size = 64
    print_out = 10

    #n_input = n_output = len(blurred_images)
    # Store layers weight & bias

    wc1 = tf.Variable(tf.random_normal([f1, f1, 3, n1], stddev=0.1), name="wc1") # 1 input, n1 outputs
    wc2 = tf.Variable(tf.random_normal([f2, f2, n1, n2], stddev=0.1), name="wc2") # n1 inputs, n2 outputs
    wc3 = tf.Variable(tf.random_normal([f3, f3, n2, 3], stddev=0.1), name="wc3") # n2 inputs, 1 outputs

    bc1 = tf.Variable(tf.random_normal(shape=[n1], stddev=0.1), name="bc1")
    bc2 = tf.Variable(tf.random_normal(shape=[n2], stddev=0.1), name="bc2")
    bc3 = tf.Variable(tf.random_normal(shape=[3], stddev=0.1), name="bc3")

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, __tile_size__, __tile_size__, 3])
    y = tf.placeholder(tf.float32, [None, __tile_size__ - _crop_size * 2, __tile_size__ - _crop_size * 2, 3])
    #y = tf.placeholder(tf.float32, [None, __tile_size__, __tile_size__, 3])


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
    def cost_func(self, pred, y, crop_size):
       slice_pred = pred[:, crop_size:__tile_size__ - crop_size, crop_size:__tile_size__ - crop_size, :]
       sub = tf.sub(slice_pred, y)
       # sum over all dimensions except batch dim
       sum = tf.reduce_sum(tf.pow(tf.sub(slice_pred, y), 2), [1, 2, 3]);
       #take mean over batch
       loss = tf.reduce_mean(sum)
       return loss
       #return tf.div(loss, (3 * (__tile_size__ - crop_size * 2) * (__tile_size__ - crop_size * 2) * self.batch_size))
    """
    def cost_func(self, pred, y, crop_size):
        i = 0
        sum = tf.Variable(0.0)
    
        #get center
        while i < self.batch_size:
            slice_y = y[i,:,:,:]
            #del conv border
            slice_pred = tf.image.resize_image_with_crop_or_pad(pred[i,:,:,:], __tile_size__ - crop_size * 2, __tile_size__ - crop_size * 2)
            tf.add(sum, tf.reduce_sum(tf.pow(tf.sub(slice_pred, slice_y), 2)))
            i = i + 1
        
        return sum
        #return tf.div(sum, (3 * (__tile_size__ - crop_size * 2) * (__tile_size__ - crop_size * 2) * self.batch_size))
    """
    def train(self, blurred_images_tiles, original_images_tiles, model_save_name):
        n_input = len(blurred_images_tiles)
        # Construct model
        count_batch = n_input / self.batch_size


        # Convolution Layer
        conv1 = self.conv2d(self.x, self.wc1, self.bc1)

        # Convolution Layer
        conv2 = self.conv2d(conv1, self.wc2, self.bc2)

        # Convolution Layer
        conv3 = self.conv2d(conv2, self.wc3, self.bc3)

        with tf.name_scope('model'):
            pred = self.model(self.x, self.wc1, self.wc2, self.wc3, self.bc1, self.bc2, self.bc3)
            # Define loss and optimizer

            # Euclidean distance
            with tf.name_scope('cost'):
                cost = self.cost_func(pred, self.y, _crop_size)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)

            with tf.name_scope('accuracy'):
                euclid = self.cost_func(pred, self.y, _crop_size)
                accuracy = tf.reduce_mean(tf.cast(tf.sqrt(euclid), tf.float32))

        # Initializing the variables
        init = tf.initialize_all_variables()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

        # Launch the graph

        # config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        with tf.Session() as sess:
            sess.run(init)
            writer = tf.train.SummaryWriter("/tmp/log", sess.graph)

            step = 1
            # Keep training until reach max iterations
            while step <= count_batch:
                #norming batches
                batch_xs_names = blurred_images_tiles[self.batch_size * (step - 1):self.batch_size * step]
                batch_ys_names = original_images_tiles[self.batch_size * (step - 1):self.batch_size * step]
                batch_xs = []
                batch_ys = []
                for i in range(0, self.batch_size, 1):
                    image = np.array(imread(batch_xs_names[i])) / 255.0
                    batch_xs.append(image)
                    image = np.array(SuperResolutionDataSet().centeredCrop(imread(batch_ys_names[i]), __tile_size__ - _crop_size * 2, __tile_size__ - _crop_size * 2)) / 255.0
                    #image = np.array(imread(batch_ys_names[i])) / 255.0
                    batch_ys.append(image)

                # Fit training using batch data
                sess.run(optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys})
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={self.x: batch_xs, self.y: batch_ys})

                if step % self.print_out == 0:
                    print "Iter " + str(step) + ", Training Accuracy = " + str(acc) + ", loss = " + str(loss)
                    c1 = sess.run(conv1, feed_dict={self.x: batch_xs})
                    c2 = sess.run(conv2, feed_dict={self.x: batch_xs})
                    c3 = sess.run(conv3, feed_dict={self.x: batch_xs})
                    print "1st conv avg:", np.average(np.array(c1)), "; 2nd conv avg:", np.average(np.array(c2)), "; 3rd conv avg:", np.average(np.array(c3))
                    print "bc3:", self.bc3.eval()
                    print "complete:", float(step) / count_batch * 100.0, "%"
                    print "_________________________________________________________"
                    print
                
                step += 1



            print "Optimization Finished!"

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
        #blurred_image = imread(image_path)
        blurred_image = [blurred_image]
        saver = tf.train.Saver()
        val = self.model(self.x, self.wc1, self.wc2, self.wc3, self.bc1, self.bc2, self.bc3)

        with tf.Session() as sess:
            saver.restore(sess, model_name)
            print("Model restored.")
            bc3 = sess.run(self.bc3)
            print "Restored last Biases:", bc3

            res = np.array(sess.run(val, feed_dict={self.x: blurred_image})) * 255
            print "output avg:", np.average(res)

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

print "start"
# main code
if (need_train):
    blurred_tiles, original_tiles = SuperResolutionDataSet().generate_dataset()
    SuperResolutionModel().train(blurred_tiles, original_tiles, "model.ckpt")

#test
if (need_evaluate):
    SuperResolutionModel().evaluate_model("model.ckpt", "test.jpg")
