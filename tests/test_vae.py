#!/usr/bin/env python

import unittest
import os

import tensorflow.python.ops.numpy_ops.np_config as np_config
np_config.enable_numpy_behavior()

from .dep import utils

class TestVAE(unittest.TestCase):

    def test_vae_get_cost(self):
        import tensorflow as tf
        import numpy as np

        img = utils.get_img()
        img = np.asarray(img) / 255.
        vae = utils.construct_vae()

        cost = vae.get_cost(tf.expand_dims(img, axis=0), utils.get_img_label_ones_hot())

        self.assertIsNotNone(cost)
        self.assertIn('loss', cost)
        self.assertIn('reconstruction_loss', cost)
        self.assertIn('kl_loss', cost)
        self.assertIn('style_loss', cost)
        self.assertIn('categorical_cross_entropy', cost)
        self.assertIn('accuracy', cost)

        expected_loss = 12988.09375
        expected_reconstruction_loss = 0.6931264400482178
        expected_kl_loss = 0.001852748915553093
        expected_style_loss = 0.00016307331679854542
        expected_cce_loss = 1.1777501106262207
        expected_accuracy = 1

        print(f'Cost: {cost}')
        print(f"Loss: {cost['loss']}")
        print(f"Reconstruction Loss: {cost['reconstruction_loss']}")
        print(f"KL Loss: {cost['kl_loss']}")
        print(f"Style Loss: {cost['style_loss']}")
        print(f"CCE Loss: {cost['categorical_cross_entropy']}")
        print(f"Accuracy: {cost['accuracy']}")
        
        self.assertAlmostEqual(expected_loss, tf.cast(cost['loss'], tf.float32), 1)
        self.assertAlmostEqual(expected_reconstruction_loss, tf.cast(cost['reconstruction_loss'], tf.float32), 1)
        self.assertAlmostEqual(expected_kl_loss, tf.cast(cost['kl_loss'], tf.float32), 1)
        self.assertAlmostEqual(expected_style_loss, tf.cast(cost['style_loss'], tf.float32), 1)
        self.assertAlmostEqual(expected_cce_loss, tf.cast(cost['categorical_cross_entropy'], tf.float32), 1)
        self.assertAlmostEqual(expected_accuracy, tf.cast(cost['accuracy'], tf.float32))

