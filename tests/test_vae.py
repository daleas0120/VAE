#!/usr/bin/env python

import unittest
import os

import tensorflow.python.ops.numpy_ops.np_config as np_config
np_config.enable_numpy_behavior()

from .dep import utils

class TestVAE(unittest.TestCase):

    def test_img_exists(self):
        img_filename = utils.get_img_filename()
        self.assertTrue(os.path.exists(img_filename))
        self.assertTrue(os.path.isfile(img_filename))

    def test_img(self):
        img = utils.get_img()
        self.assertIsNotNone(img)
        self.assertEqual(img.size[0], utils.get_img_dim())
        self.assertEqual(img.size[1], utils.get_img_dim())

    def test_vae_get_cost(self):
        import tensorflow as tf
        import numpy as np

        img = utils.get_img()
        img = np.asarray(img) / 255.
        vae = utils.construct_vae()

        cost = vae.get_cost(tf.expand_dims(img, axis=0))

        self.assertIsNotNone(cost)
        self.assertIn('loss', cost)
        self.assertIn('reconstruction_loss', cost)
        self.assertIn('kl_loss', cost)
        self.assertIn('style_loss', cost)

        expected_loss = 12986.9140625
        expected_reconstruction_loss = 0.6931264400482178
        expected_kl_loss = 0.001852748915553093
        expected_style_loss = 0.00016307331679854542

        print(f'Cost: {cost}')
        print(f"Loss: {cost['loss']}")
        print(f"Reconstruction Loss: {cost['reconstruction_loss']}")
        print(f"KL Loss: {cost['kl_loss']}")
        print(f"Style Loss: {cost['style_loss']}")

        self.assertAlmostEqual(expected_loss, tf.cast(cost['loss'], tf.float32), 1)
        self.assertAlmostEqual(expected_reconstruction_loss, tf.cast(cost['reconstruction_loss'], tf.float32), 1)
        self.assertAlmostEqual(expected_kl_loss, tf.cast(cost['kl_loss'], tf.float32), 1)
        self.assertAlmostEqual(expected_style_loss, tf.cast(cost['style_loss'], tf.float32), 1)
