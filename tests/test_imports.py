#!/usr/bin/env python

import unittest

class TestImports(unittest.TestCase):
    def test_import_tf(self):
        import tensorflow
        self.assertIsNotNone(tensorflow)

    def test_import_keras(self):
        import tensorflow.keras as keras
        self.assertIsNotNone(keras)

    def test_tf_gpu(self):
        import tensorflow as tf
        num_gpu = len(tf.config.list_physical_devices('GPU'))
        self.assertNotEqual(num_gpu, 0)

    def test_import_vae_package(self):
        import VAE
        self.assertIsNotNone(VAE)

    def test_import_vae_module(self):
        from VAE import VAE
        self.assertIsNotNone(VAE)
        