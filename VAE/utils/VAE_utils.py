#!/usr/bin/env python

import json
import os
import time
import tqdm
from tqdm import trange
import shutil
from multiprocessing import Pool
from itertools import repeat
import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras

import skimage as sk
import skimage.io

from VAE.utils import utils as ut
from VAE.utils import custom_parse_rgb as parse_txt


def load_image_as_array(img_path: str) -> np.ndarray:
    assert(os.path.exists(img_path))
    return np.array(skimage.io.imread(img_path))

def load_single_image(annotation, img_dim, img_ch, output_preprocess_path):
    try:
        # Load img set into memory.  This is only manageable since the dataset is tiny.
        image = load_image_as_array(annotation['img_path'])

        if img_ch > 1:
            if len(image.shape) == 2:
                from skimage.color import gray2rgb
                image = gray2rgb(image)

        height, width = image.shape[:2]

        max_pixel = np.max(image)
        min_pixel = np.min(image)
        image = (image - min_pixel) / (max_pixel - min_pixel)

        if (height != img_dim) or (width != img_dim):
                image = sk.transform.resize(image[:, :, :img_ch], (img_dim, img_dim, img_ch))

        if annotation['z_filename'] != 'None':
            z_image = load_image_as_array(annotation['z_img_path'])

            height, width = z_image.shape[:2]

            if (height != img_dim) or (width != img_dim):
                z_image = sk.transform.resize(z_image[:, :, 1], (img_dim, img_dim, 1))

            image = np.dstack((image, z_image))

        # Write out the preprocessed image
        if output_preprocess_path is not None:
            # Note: convert to uint8 to prevent warning - wmb
            skimage.io.imsave(os.path.join(output_preprocess_path, annotation['filename']), (255.*image).astype('uint8'))
    except:
        print(annotation['img_path'])
        raise

    return image, annotation['Class'], annotation['DataType']

def load_single_image_star(args):
    return load_single_image(*args)


def parse_coco_labels(coco_json, IMG_DIM, IMG_CH):
    labels = []
    imgs = []

    [path, head] = os.path.split(coco_json)
    coco_annotations = json.load(open(coco_json))
    annotations = list(coco_annotations['annotations'])
    img_list = list(coco_annotations['images'])

    for idx, a in tqdm.tqdm(enumerate(img_list), desc='Loading Images', total=len(annotations)):
        try:
            # Load img set into memory.  This is only manageable since the dataset is tiny.

            img_path = os.path.join(path, a['file_name'])
            image = np.array(skimage.io.imread(img_path))
            height, width = image.shape[:2]

            if (height != IMG_DIM) or (width != IMG_DIM):
                image = sk.transform.resize(image[:, :, :IMG_CH], (IMG_DIM, IMG_DIM, IMG_CH))

            imgs.append(image)

            img_id = a['id']
            class_id = annotations[img_id-1]['category_id']
            superclass_name = coco_annotations['categories'][class_id]['supercategory']
            class_name = coco_annotations['categories'][class_id]['name']
            labels.append([superclass_name, class_name])

            img_list[idx]['Class'] = class_name
            img_list[idx]['SuperClass'] = superclass_name

        except:
            print(img_path)
            raise

    return np.asarray(imgs), labels, img_list


class RGB_Dataset(ut.Dataset):

    def load_rgb(self, IMG_DIM, IMG_CH, dataset_dir=None, groundTruthFile=None, output_preprocess_path=None, workers=8):

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
            return parse_coco_labels(datafile, IMG_DIM, IMG_CH)

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
        
        # Multiprocessing version
        r_val = list()
        pool_args = zip(annotations, repeat(IMG_DIM), repeat(IMG_CH), repeat(output_preprocess_path))
        with Pool(workers) as pool:
            r_val = list(tqdm.tqdm(pool.imap(load_single_image_star, pool_args), total=len(annotations), desc='Loading Images'))

        for r in r_val:
            imgs.append(r[0])
            labels.append([r[1], r[2]])
        

        """
        # Use this for single-process
        for annotation in tqdm.tqdm(annotations, desc='Loading Images'):
            image, class_name, data_type = load_single_image(annotation, IMG_DIM, IMG_CH, output_preprocess_path)
            labels.append([class_name, data_type])
            imgs.append(image)
        """

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

def get_style_loss(style, combination, img_nrows, img_ncols):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def get_reconstruction_error(img_tensor, reconstruction_tensor):
    return tf.reduce_mean(
        keras.losses.binary_crossentropy(img_tensor, reconstruction_tensor)
    )

def get_kl_loss(z_mean, z_log_var):
    return -tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    )
    

def customLoss(data, featuresOG, features, IMG_DIM):

    content_weight = IMG_DIM*IMG_DIM
    style_weight = 1e2
    kl_weight = -0.5

    z_meanOG = featuresOG['z_mean']
    z_log_varOG = featuresOG['z_log_var']
    reconstruction = featuresOG['conv1decoder']

    d = tf.cast(data, dtype=tf.float32)

    # Get content/reconstruction loss
    content_loss = get_reconstruction_error(d, reconstruction)
    content_loss *= content_weight

    # Get style losses

    # Encoded vs Decoded Img Activations in the Encoder
    sl1 = get_style_loss(featuresOG['conv1encoder'], features['conv1encoder'], IMG_DIM, IMG_DIM)
    sl2 = get_style_loss(featuresOG['conv2encoder'], features['conv2encoder'], IMG_DIM, IMG_DIM)
    sl3 = get_style_loss(featuresOG['conv3encoder'], features['conv3encoder'], IMG_DIM, IMG_DIM)

    # Encoder vs Decoder Activations
    sl4 = get_style_loss(featuresOG['conv1encoder'], featuresOG['conv2decoder'], IMG_DIM, IMG_DIM)
    sl5 = get_style_loss(featuresOG['conv2encoder'], featuresOG['conv3decoder'], IMG_DIM, IMG_DIM)
    sl6 = get_style_loss(featuresOG['conv3encoder'], featuresOG['reshape_1'], IMG_DIM, IMG_DIM)

    sL = tf.reduce_mean((style_weight / 6) * (sl1 + sl2 + sl3 + sl4 + sl5 + sl6))

    # Get KL Loss
    kl_loss = 1 + z_log_varOG - tf.square(z_meanOG) - tf.exp(z_log_varOG)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= kl_weight

    total_loss = content_loss + kl_loss

    return total_loss, content_loss, kl_loss,  sL

def getImageLoss(gt_img, img):
    reconstruction_loss = tf.reduce_mean(
        keras.losses.binary_crossentropy(gt_img[None, :, :, :], img[None, :, :, :])
    )
    #reconstruction_loss *= IMG_DIM * IMG_DIM

    return reconstruction_loss


def dimension_wise_KLdivergence():
    return 0

def total_correlation():
    return 0

def index_code_mutual_information():
    return 0

def get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    ## This is a straight up copy of Yann Dubs code into tensorflow.
    ## https://github.com/YannDubs/disentangling-vae/blob/master/disvae/models/losses.py

    batch_size, hidden_dim = latent_sample.shape

    #calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)


def disentangle_loss(data, featuresOG, features, IMG_DIM, a=1, b=1, c=1):

    reconstruction_loss = customLoss(data, featuresOG, features, IMG_DIM)

    mi_loss = index_code_mutual_information()

    tc_loss = total_correlation()

    dw_kl_loss = dimension_wise_KLdivergence()

    loss = reconstruction_loss + a*mi_loss + b*tc_loss + c*dw_kl_loss

    return loss


def generate_img_set(decoder, z, filepath, orig_list, orig_imgs, IMG_DIM, IMG_CH):
    print("Generating Latent Space Image Set")
    loss_per_img = []
    path = filepath + "/generatedImgs/"
    os.mkdir(path)
    num_imgs = len(orig_list)
    for i in range(num_imgs):
        gt_img = orig_imgs[i]
        img_name = orig_list[i]['file_name']

        # plt.figure()
        # ax = plt.axes([0, 0, 1, 1], frameon=False)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.imshow(gt_img)
        # plt.autoscale(tight=True)
        # plt.show()

        #img = decoder(np.array([z[i]]))
        _, _, _, _, img = decoder(np.array([z[i]]))
        img = np.array(img)
        img = img[0, :, :, :]
        reconstruction_loss = getImageLoss(gt_img, img)

        # ---Format to write out in 128x128 img format with openCV---#

        img_rgb = 255.999 * img
        img_rgb = np.array(img_rgb, dtype='float32').astype('uint8')
        img_bgr = np.dstack((img_rgb[:, :, 2], img_rgb[:, :, 1], img_rgb[:, :, 0]))

        cv2.imwrite(os.path.join(filepath, "generatedImgs", img_name), img_bgr)

        # img = img[0].reshape(IMG_DIM, IMG_DIM, IMG_CH)

        # plt.figure()
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        # plt.close()

        loss_per_img.append(float(np.array(reconstruction_loss)))

    return loss_per_img