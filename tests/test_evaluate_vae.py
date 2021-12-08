#!/usr/bin/env python

import unittest
import os

import numpy as np
from numpy.testing import assert_array_equal

from .dep import utils
_ = utils.construct_vae()

class TestEvaluateVAE(unittest.TestCase):

    def test_load_encoder(self):
        from evaluate_vae import load_encoder
        encoder = load_encoder(utils.get_model_path())
        self.assertIsNotNone(encoder)

    def test_load_decoder(self):
        from evaluate_vae import load_decoder
        decoder = load_decoder(utils.get_model_path())
        self.assertIsNotNone(decoder)

    def test_load_model(self):
        from evaluate_vae import load_model
        encoder, decoder = load_model(utils.get_model_path(), utils.get_model_path())
        self.assertIsNotNone(encoder)
        self.assertIsNotNone(decoder)

    def test_normalize_images(self):
        from evaluate_vae import normalize_images
        from VAE.utils.VAE_utils import load_image_as_array

        imgs = np.array([load_image_as_array(utils.get_img_filename())])
        norm_imgs = normalize_images(imgs)

        self.assertEqual(norm_imgs.shape, (1, 128, 128, 3))
        self.assertAlmostEqual(np.sum(norm_imgs), 19975.718253968254)


    def test_load_data(self):
        pass

    def test_generate_latent_space(self):
        pass

    def test_write_latent_space(self):
        pass

    ## After moving internal functions

    def test_generate_img_set(self):
        pass