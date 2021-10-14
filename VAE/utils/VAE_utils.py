#!/usr/bin/env python

import json
import os
import time
import tqdm
from tqdm import trange
import shutil

import numpy as np

import tensorflow as tf
from tensorflow import keras

import skimage as sk
import skimage.io

from VAE.utils import utils as ut
from VAE.utils import custom_parse_rgb as parse_txt


class RGB_Dataset(ut.Dataset):

    def load_rgb(self, IMG_DIM, IMG_CH, dataset_dir=None, groundTruthFile=None, output_preprocess_path=None):

        labels = []
        imgs = []

        if dataset_dir == None and groundTruthFile == None:
            raise ValueError("ERROR: No Data Found")

        if groundTruthFile is None:
            print("No GroundTruthFile specified")
            # search for .txt or .json and return groundTruth File
            for file in os.listdir(dataset_dir):
                if file.endswith(".txt"):
                    print("GroundTruthFile Found: ", file)
                    groundTruthFile = file
                    datafile = os.path.join(dataset_dir, groundTruthFile)
                    break
        if groundTruthFile is None:
            for file in os.listdir(dataset_dir):
                if file.endswith('csv'):
                    print("GroundTruthFile Founds: ", file)
                    groundTruthFile = file
                    datafile = os.path.join(dataset_dir, groundTruthFile)
                    break
        if groundTruthFile is None: #no .txt file found
            for file in os.listdir(dataset_dir):
                if file.endswith(".json"):
                    print("GroundTruthFile Found: ", file)
                    groundTruthFile = file
                    datafile = os.path.join(dataset_dir, groundTruthFile)
                    break
        else:
            datafile = groundTruthFile

        print("datafile: ", datafile)

        # check to see if groundTruthFile is .txt or .json
        extension = os.path.splitext(groundTruthFile)[1]
        if extension == ".json":
            annotations = json.load(open(datafile))
        elif extension == ".txt":  # load txt file
            annotations, dataset_dir = parse_txt.load(datafile)
            print('Number of images found: ', len(annotations))
        elif extension == '.csv':
            annotations, dataset_dir = parse_txt.load_csv(datafile)

        else:
            raise IOError("ERROR: GroundTruthFile type is not .txt or .json")

        # Create output directory for preprocessed images

        if output_preprocess_path is not None:
            if os.path.exists(output_preprocess_path):
                assert(os.path.isdir(output_preprocess_path))
                shutil.rmtree(output_preprocess_path)
            os.makedirs(output_preprocess_path)

        annotations = list(annotations.values())

        # Add images

        for idx,a in tqdm.tqdm(enumerate(annotations), desc='Loading Images', total=len(annotations)):
            try:
                # Load img set into memory.  This is only manageable since the dataset is tiny.
                image = np.array(skimage.io.imread(a['img_path']))

                if IMG_CH > 1:
                    if len(image.shape) == 2:
                        from skimage.color import gray2rgb
                        image = gray2rgb(image)


                height, width = image.shape[:2]

                if (height != IMG_DIM) or (width != IMG_DIM):
                        image = sk.transform.resize(image[:, :, :IMG_CH], (IMG_DIM, IMG_DIM, IMG_CH))

                if a['z_filename'] != 'None':
                    z_image = np.array(skimage.io.imread(a['z_img_path']))

                    height, width = z_image.shape[:2]

                    if (height != IMG_DIM) or (width != IMG_DIM):
                        z_image = sk.transform.resize(z_image[:, :, 1], (IMG_DIM, IMG_DIM, 1))

                    image = np.dstack((image, z_image))

                # Write out the preprocessed image
                if output_preprocess_path is not None:
                    skimage.io.imsave(os.path.join(output_preprocess_path, a['filename']), image)

                labels.append([a['Class'], a['DataType']])
                imgs.append(image)

            #time.sleep(0.1)
            except:
                print(a['img_path'])
                raise

        return np.asarray(imgs), labels, annotations


# The gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):

    x = tf.transpose(x, perm=[2, 0, 1, 3])

    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


# The "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image

def style_loss(style, combination, img_nrows, img_ncols):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def reconstruction_error(img_tensor, reconstruction_tensor):
    return tf.reduce_mean(tf.math.abs(
        keras.losses.binary_crossentropy(img_tensor, reconstruction_tensor)
    ))
    

def customLoss(data, featuresOG, features, IMG_DIM):

    content_weight = IMG_DIM*IMG_DIM
    style_weight = 1e2
    kl_weight = -0.5

    z_meanOG = featuresOG['z_mean']
    z_log_varOG = featuresOG['z_log_var']
    reconstruction = featuresOG['conv1decoder']

    d = tf.cast(data, dtype=tf.float32)

    # Get content/reconstruction loss
    content_loss = reconstruction_error(d, reconstruction)
    content_loss *= content_weight

    # Get style losses

    # Encoded vs Decoded Img Activations in the Encoder
    sl1 = style_loss(featuresOG['conv1encoder'], features['conv1encoder'], IMG_DIM, IMG_DIM)
    sl2 = style_loss(featuresOG['conv2encoder'], features['conv2encoder'], IMG_DIM, IMG_DIM)
    sl3 = style_loss(featuresOG['conv3encoder'], features['conv3encoder'], IMG_DIM, IMG_DIM)

    # Encoder vs Decoder Activations
    sl4 = style_loss(featuresOG['conv1encoder'], featuresOG['conv2decoder'], IMG_DIM, IMG_DIM)
    sl5 = style_loss(featuresOG['conv2encoder'], featuresOG['conv3decoder'], IMG_DIM, IMG_DIM)
    sl6 = style_loss(featuresOG['conv3encoder'], featuresOG['reshape_1'], IMG_DIM, IMG_DIM)

    sL = tf.reduce_mean((style_weight / 6) * (sl1 + sl2 + sl3 + sl4 + sl5 + sl6))

    # Get KL Loss
    kl_loss = 1 + z_log_varOG - tf.square(z_meanOG) - tf.exp(z_log_varOG)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= kl_weight

    total_loss = content_loss + kl_loss

    return total_loss, content_loss, kl_loss,  sL

