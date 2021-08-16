#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras

from VAE.utils.VAE_utils import style_loss

"""
## Define the VAE as a `Model` with a custom `train_step`
"""

class VAE(keras.Model):
    def __init__(self, encoder, decoder, content_weight, style_weight, kl_weight, img_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.kl_weight = kl_weight
        self.img_dim = img_dim


    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:

            # Send the image through the encoder, return conv layer activations and latent space embedding
            [ec1_0, ec2_0, ec3_0, _, _, z_mean_0, z_log_var_0, z_0] = self.encoder(data)

            # Decode latent space representation to obtain reconstruction image and conv layer activations
            [_, reshape, dc3, dc2, reconstruction] = self.decoder(z_0)

            # Encode the reconstruction to compare conv layer activations
            [ec1_1, ec2_1, ec3_1, _, _, _, _, _] = self.encoder(reconstruction)

            # Reconstruction loss is BCE
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= self.content_weight

            # Latent Space Loss us Kullback-Leibler Divergence
            kl_loss = 1 + z_log_var_0 - tf.square(z_mean_0) - tf.exp(z_log_var_0)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= self.kl_weight

            # Style loss is gram matrix comparison of matched encoder and decoder layers for the
            # original input image and the reconstructed image

            # Encoded vs Decoded Img Activations in the Encoder
            sl1 = style_loss(ec1_0, ec1_1, self.img_dim, self.img_dim)
            sl2 = style_loss(ec2_0, ec2_1, self.img_dim, self.img_dim)
            sl3 = style_loss(ec3_0, ec3_1, self.img_dim, self.img_dim)

            # Encoder vs Decoder Activations for the original image
            sl4 = style_loss(ec1_0, dc2, self.img_dim, self.img_dim)
            sl5 = style_loss(ec2_0, dc3, self.img_dim, self.img_dim)
            sl6 = style_loss(ec3_0, reshape, self.img_dim, self.img_dim)

            sL = tf.reduce_mean((self.style_weight / 6) * (sl1 + sl2 + sl3 + sl4 + sl5 + sl6))

            total_loss = reconstruction_loss + kl_loss + sL

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "style_loss": sL,
        }


    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        [ec1_0, ec2_0, ec3_0, _, _, z_mean_0, z_log_var_0, z_0] = self.encoder(data)

        [_, reshape, dc3, dc2, reconstruction] = self.decoder(z_0)

        [ec1_1, ec2_1, ec3_1, _, _, _, _, _] = self.encoder(reconstruction)

        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(data, reconstruction)
        )
        reconstruction_loss *= self.content_weight

        kl_loss = 1 + z_log_var_0 - tf.square(z_mean_0) - tf.exp(z_log_var_0)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= self.kl_weight

        # Get style losses

        # Encoded vs Decoded Img Activations in the Encoder
        sl1 = style_loss(ec1_0, ec1_1, self.img_dim, self.img_dim)
        sl2 = style_loss(ec2_0, ec2_1, self.img_dim, self.img_dim)
        sl3 = style_loss(ec3_0, ec3_1, self.img_dim, self.img_dim)

        # Encoder vs Decoder Activations
        sl4 = style_loss(ec1_0, dc2, self.img_dim, self.img_dim)
        sl5 = style_loss(ec2_0, dc3, self.img_dim, self.img_dim)
        sl6 = style_loss(ec3_0, reshape, self.img_dim, self.img_dim)

        sL = tf.reduce_mean((self.style_weight / 6) * (sl1 + sl2 + sl3 + sl4 + sl5 + sl6))

        total_loss = reconstruction_loss + kl_loss + sL

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "style_loss": sL,
        }
        