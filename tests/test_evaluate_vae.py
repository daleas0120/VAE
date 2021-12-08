#!/usr/bin/env python

import unittest
import os

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
        pass

    def test_load_data(self):
        pass

    def test_generate_latent_space(self):
        pass

    def test_write_latent_space(self):
        pass

    ## After moving internal functions

    def test_generate_img_set(self):
        pass