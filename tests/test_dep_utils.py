#!/usr/bin/env python

import unittest
import os

from .dep import utils
_ = utils.construct_vae()

class TestDepUtils(unittest.TestCase):
    def test_get_mock_dir(self):
        mock_dir = utils.get_mock_dir()
        self.assertTrue(os.path.exists(mock_dir))
        self.assertTrue(os.path.isdir(mock_dir))

    def test_model_path(self):
        model_path = utils.get_model_path()
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.isdir(model_path))

    def test_encoder_path(self):
        encoder_path = utils.get_encoder_path()
        self.assertTrue(os.path.exists(encoder_path))
        self.assertTrue(os.path.isdir(encoder_path))

    def test_decoder_path(self):
        decoder_path = utils.get_decoder_path()
        self.assertTrue(os.path.exists(decoder_path))
        self.assertTrue(os.path.isdir(decoder_path))