#!/usr/bin/env python

import unittest
import os

def get_seed():
    return 12345

def get_img_dim():
    return 128

def get_img_channels():
    return 3

def get_latent_dim():
    return 32

def get_weight_bce():
    img_dim = get_img_dim()
    return img_dim*img_dim

def get_weight_sl():
    return 1E7

def get_weight_kl():
    return -0.5

def construct_vae():
    from VAE.include.VAE_arch import encoder_3conv7c, decoder_3conv7c
    from VAE.VAE import VAE
    encoder = encoder_3conv7c(get_img_dim(), get_img_channels(), get_latent_dim())
    decoder = decoder_3conv7c(get_latent_dim())
    vae = VAE(encoder, decoder, get_weight_bce(), get_weight_sl(), get_weight_kl(), get_img_dim())
    return vae

def get_img_filename():
    mock_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mock_data'))
    return os.path.join(mock_data_dir, 'preprocessed_img_128x128x3.png')

def get_img():
    from PIL import Image
    img = Image.open(get_img_filename())
    return img

class TestVAE(unittest.TestCase):

    def test_img_exists(self):
        img_filename = get_img_filename()
        self.assertTrue(os.path.exists(img_filename))
        self.assertTrue(os.path.isfile(img_filename))

    def test_img(self):
        img = get_img()
        self.assertIsNotNone(img)
        self.assertEqual(img.size[0], get_img_dim())
        self.assertEqual(img.size[1], get_img_dim())

    def test_vae_get_cost(self):
        import tensorflow as tf
        import numpy as np

        np.random.seed(get_seed())
        tf.random.set_seed(get_seed())

        img = get_img()
        vae = construct_vae()

        cost = vae.get_cost(tf.expand_dims(np.asarray(img), axis=0))

        self.assertIsNotNone(cost)
        self.assertIn('loss', cost)
        self.assertIn('reconstruction_loss', cost)
        self.assertIn('kl_loss', cost)
        self.assertIn('style_loss', cost)

        expected_loss = 9542534430720.0
        expected_reconstruction_loss = 312102.59375
        expected_kl_loss = 1919598.0
        expected_style_loss = 9542532333568.0

        print(f'Cost: {cost}')
        print(f"Loss: {cost['loss']}")
        print(f"Reconstruction Loss: {cost['reconstruction_loss']}")
        print(f"KL Loss: {cost['kl_loss']}")
        print(f"Style Loss: {cost['style_loss']}")

        self.assertAlmostEqual(expected_loss, cost['loss'].numpy())
        self.assertAlmostEqual(expected_reconstruction_loss, cost['reconstruction_loss'].numpy(), 3)
        self.assertAlmostEqual(expected_kl_loss, cost['kl_loss'].numpy(), 3)
        self.assertAlmostEqual(expected_style_loss, cost['style_loss'].numpy(), 3)
