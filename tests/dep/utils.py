#!/usr/bin/env python

import os
import tensorflow as tf
import numpy as np

def get_mock_dir():
    curdir = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(curdir, '..', 'mock_data'))

## Mock Model Utils

def get_model_path():
    mock_dir = get_mock_dir()
    return os.path.join(mock_dir, 'model')

def get_encoder_path():
    model_path = get_model_path()
    return os.path.join(model_path, 'encoder')

def get_decoder_path():
    model_path = get_model_path()
    return os.path.join(model_path, 'decoder')

def get_classifier_path():
    model_path = get_model_path()
    return os.path.join(model_path, 'classifier')

def get_seed():
    return 12345

def set_seed():
    np.random.seed(get_seed())
    tf.random.set_seed(get_seed())

def get_img_dim():
    return 128

def get_img_channels():
    return 3

def get_latent_dim():
    return 32

def get_class_size():
    return 5

def get_weight_bce():
    img_dim = get_img_dim()
    return img_dim*img_dim

def get_weight_sl():
    return 1E7

def get_weight_kl():
    return -0.5

def get_weight_class():
    return 1.0

def construct_vae():
    from VAE.include.VAE_arch import encoder_3conv7c, decoder_3conv7c, latent_classifier_arch
    from VAE.VAE import VAE
    set_seed()
    encoder = encoder_3conv7c(get_img_dim(), get_img_channels(), get_latent_dim())
    decoder = decoder_3conv7c(get_latent_dim())
    classifier = latent_classifier_arch(get_latent_dim(), get_class_size())
    vae = VAE(encoder, decoder, classifier, get_weight_bce(), get_weight_sl(), get_weight_kl(), get_weight_class(), get_img_dim())

    if not os.path.exists(get_encoder_path()):
        encoder.save(filepath=get_encoder_path())
    if not os.path.exists(get_decoder_path()):
        decoder.save(filepath=get_decoder_path())
    if not os.path.exists(get_classifier_path()):
        classifier.save(filepath=get_classifier_path())

    return vae

## Mock Image

def get_img_filename():
    mock_data_dir = get_mock_dir()
    return os.path.join(mock_data_dir, 'preprocessed_img_128x128x3.png')

def get_img():
    from PIL import Image
    img = Image.open(get_img_filename())
    return img

def get_img_label():
    return ('Plane', 'RW')

def get_img_label_ones_hot():
    return [[0, 0, 0, 1, 0]]

