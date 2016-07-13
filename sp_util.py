"""
author: Vladimir Mikhelev 2016
utils for pre-processing images for Super Resolution method
"""

# coding=utf-8
import ntpath
import os
import warnings

import numpy as np
import skimage
import skimage.transform
import tensorflow as tf
from skimage.io import imsave, imread

#constants sections
__need_blurred_image__ = 0          # flag to generate blurred image
__upscale_coefficient__ = 2         # coefficient for image blurring
_original_image_directory = "images"
_original_tiles_directory = "out/original_tiles"
_blurred_tiles_directory = "out/blurred_tiles"

#open directory with images
def get_directory(folder):
    foundfile = []
    for path, subdirs, files in os.walk(folder):
        for name in files:
            found = os.path.join(path, name)
            if name.endswith('.jpg'):
                foundfile.append(found)

    foundfile.sort()
    return foundfile

#blurred image as <_coefficient> downscale then <_coefficient> upscale
def get_blurred_image(_path, _coefficient):
    img = imread(_path)
    img = skimage.transform.rescale(img, 1.0/_coefficient)
    img = skimage.transform.rescale(img, _coefficient)
    return img


def rgb2gray(rgb):
    img_h = rgb.shape[0]
    img_w = rgb.shape[1]

    gray = np.zeros([img_h, img_w, 1])

    for h in range(img_h):
        for w in range(img_w):
            gray[h, w, 0] = 0.2989 * rgb[h, w, 0] + 0.5870 * rgb[h, w, 1] + 0.1140 * rgb[h, w, 2]

    return gray


# cpu impl
def read_image(filenames, channels, need_crop, tile_size, crop_size):
    images = []
    for i in range(0, len(filenames), 1):
        image = np.array(imread(filenames[i]) / 255.0)

        # grayscale
        if channels == 1:
            image = rgb2gray(image)

        if need_crop == 1:
            image = np.array(centeredCrop(image, tile_size - crop_size * 2, tile_size - crop_size * 2))
        images.append(image)
    return np.array(images)

#do tiles tf version - slow
def crop_tiles_in_folder_tf(_path, _image, height, width, image_name, tile_size, crop_size):
    i = 0
    with tf.Session() as sess:
        for h in range(0, height, tile_size):
            for w in range(0, width, tile_size):
                image = tf.image.crop_to_bounding_box(_image, h, w, tile_size, tile_size)
                image = image.eval()
                imsave(_path + "/" + os.path.splitext(ntpath.basename(image_name))[0] + '_tiles_{0}.jpg'.format(i), image)
                i += 1

# tensorflow implementation - very slow  NEED FOR EVALUATE
def read_image_tf(filenames, channels, need_crop, tile_size, crop_size):
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    key, file = reader.read(filename_queue)
    uint8image = tf.image.decode_jpeg(file, channels=channels)
    if need_crop:
        uint8image = tf.image.resize_image_with_crop_or_pad(uint8image, tile_size - crop_size * 2,
                                                                        crop_size - crop_size * 2)

    float_image = tf.div(tf.cast(uint8image, tf.float32), 255)
    images = []
    with tf.Session() as sess:
        # Required to get the filename matching to run.
        tf.initialize_all_variables().run()
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(len(filenames)):
            # Get an image tensor and print its value.
            image_tensor = float_image.eval()
            images.append(image_tensor)
        coord.request_stop()
        coord.join(threads)

    return np.array(images)

#do tiles cpu version
def crop_tiles_in_folder(_path, _image, image_name, _tile_size):
    i = 0
    for h in range(0, _image.shape[0], _tile_size):
        for w in range(0, _image.shape[1], _tile_size):
            w_end = w + _tile_size
            h_end = h + _tile_size
            imsave(_path + "/" + os.path.splitext(ntpath.basename(image_name))[0] + '_tiles_{0}.jpg'.format(i), _image[w:w_end, h:h_end])
            i += 1

#clear directory
def clear_directory(_path):
    for path, subdirs, files in os.walk(_path):
        for name in files:
            found = os.path.join(path, name)
            os.unlink(found)

#crop or pad image for needly resolution from center of image
def crop_or_pad(_image, height, width):
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
        pad_img = centeredCrop(_image, height, width)

    if (img_h < height and img_w > width):
        crimg = centeredCrop(_image, height, width)
        pad_img[pad_h: -pad_top, :, :] = crimg

    if (img_h > height and img_w < width):
        crimg = centeredCrop(_image, height, width)
        pad_img[:, pad_w: -pad_right, :] = crimg

    return pad_img

#crop from center
def centeredCrop(_image, height, width):
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

#generate dataset from exsisting images, do tiles
#return blured, original tiles names
def generate_dataset(image_resize_height, image_resize_width, tile_size):
    #original images
    original_images = get_directory(_original_image_directory)
    blurred_images_tiles = []
    original_images_tiles = []
    if __need_blurred_image__:
        print "clear original tiles folder"
        clear_directory(_original_tiles_directory)
        print "clear blurred tiles folder"
        clear_directory(_blurred_tiles_directory)
        for image in original_images:
            print (image)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image_name = image
                original_image = imread(image)
                blurred_image = get_blurred_image(image, __upscale_coefficient__)
                shape = original_image.shape
                #for cpu

                if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                    blurred_image = crop_or_pad(blurred_image, image_resize_height, image_resize_width)
                    original_image = crop_or_pad(original_image, image_resize_height, image_resize_width)
                    # crop tiles
                    crop_tiles_in_folder(_blurred_tiles_directory, blurred_image, image_name, tile_size)
                    crop_tiles_in_folder(_original_tiles_directory, original_image, image_name, tile_size)
                    print shape, "->", original_image.shape

                    """
                    #slow tensorflow implementation
                    #TODO make faster variant -> tf.while_loop
                    # for gpu
                    blurred_image = tf.image.resize_image_with_crop_or_pad(blurred_image, _image_resize_height, _image_resize_width)
                    original_image = tf.image.resize_image_with_crop_or_pad(original_image, _image_resize_height, _image_resize_width)

                    # crop tiles
                    self.crop_tiles_in_folder_tf(_blurred_tiles_directory, blurred_image, _image_resize_height, _image_resize_width, image_name, __tile_size__)
                    self.crop_tiles_in_folder_tf(_original_tiles_directory, original_image, _image_resize_height, _image_resize_width, image_name, __tile_size__)
                    print shape, "->", original_image.get_shape()
                    """
    # blurred images tiles
    blurred_images_tiles = get_directory(_blurred_tiles_directory)
    original_images_tiles= get_directory(_original_tiles_directory)
    print str(len(blurred_images_tiles)) + "x2", "tiles saved"
    """
    #load all tiles in array -> VERY BAD MEMORY USAGE
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
