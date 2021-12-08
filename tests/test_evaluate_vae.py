#!/usr/bin/env python

import unittest
import os

def get_mock_dir():
    curdir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(curdir, 'mock_data')

def get_model_path():
    mock_dir = get_mock_dir()
    return os.path.join(mock_dir, 'model', '20211203T1145')

def get_encoder_path():
    model_path = get_model_path()
    return os.path.join(model_path, 'encoder')

def get_decoder_path():
    model_path = get_model_path()
    return os.path.join(model_path, 'decoder')


class TestEvaluateVAE(unittest.TestCase):

    def test_get_mock_dir(self):
        mock_dir = get_mock_dir()
        self.assertTrue(os.path.exists(mock_dir))
        self.assertTrue(os.path.isdir(mock_dir))

    def test_model_path(self):
        model_path = get_model_path()
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.isdir(model_path))

    def test_encoder_path(self):
        encoder_path = get_encoder_path()
        self.assertTrue(os.path.exists(encoder_path))
        self.assertTrue(os.path.isdir(encoder_path))

    def test_decoder_path(self):
        decoder_path = get_decoder_path()
        self.assertTrue(os.path.exists(decoder_path))
        self.assertTrue(os.path.isdir(decoder_path))
    
    def test_load_encoder(self):
        from evaluate_vae import load_encoder
        encoder = load_encoder(get_model_path())
        self.assertIsNotNone(encoder)

    def test_load_decoder(self):
        from evaluate_vae import load_decoder
        decoder = load_decoder(get_model_path())
        self.assertIsNotNone(decoder)

    def test_load_model(self):
        from evaluate_vae import load_model
        encoder, decoder = load_model(get_model_path(), get_model_path())
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