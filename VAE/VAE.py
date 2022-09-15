#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras

from VAE.utils.VAE_utils import get_style_loss, get_reconstruction_error, get_kl_loss

"""
## Define the VAE as a `Model` with a custom `train_step`
"""

class VAE(keras.Model):
    """
    Variational AutoeEncoder class, subclassed from keras.Model

    :param encoder: A handle to the Encoder model
    :type encoder: keras.Model
    :param decoder: A handle to the Decoder model
    :type decoder: keras.Model
    :param classifier: A handle to the Classifier model (can be None)
    :type classifier: keras.Model
    :param content_weight: Weight applied to Binary Cross-Entropy Reconstruction Loss
    :type content_weight: float
    :param style_weight: Weight applied to the Style Losss
    :type style_weight: float
    :param kl_weight: Weight applied to the KL-Divergence Loss of the latent vector
    :type kl_weight: float
    :param img_dim: Single square dimension of input images
    :type img_dim: int
    :param **kwargs: Arguments for keras.Model
    """
    def __init__(self, encoder, decoder, classifier, content_weight, style_weight, kl_weight, class_weight, img_dim, **kwargs):
        """
        Constructor Method
        """
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier    # Optional: provide None to disable
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.kl_weight = kl_weight
        self.class_weight = class_weight
        self.img_dim = img_dim
        self.class_accuracy = keras.metrics.Accuracy()


    def get_cost(self, x, y):
        # Send the image through the encoder, return conv layer activations and latent space embedding
        #[ec1_0, ec2_0, ec3_0, _, _, z_mean_0, z_log_var_0, z_0] = self.encoder(x)
        [c1_1e0, c1_2e0, c2_1e0, c2_2e0, c3_1e0, c3_2e0, c3_3e0, c4_1e0, c4_2e0, c4_3e0,
         c5_1e0, c5_2e0, c5_3e0, _, _, z_mean, z_log_var, z] = self.encoder(x)

        # Decode latent space representation to obtain reconstruction image and conv layer activations
        # [_, reshape, dc3, dc2, reconstruction] = self.decoder(z)
        [_, reshape, c5_3d, c5_2d, c5_1d, c4_3d, c4_2d, c4_1d, c3_3d, c3_2d,
                                          c3_1d, c2_2d, c2_1d, c1_2d, reconstruction] = self.decoder(z)

        # Encode the reconstruction to compare conv layer activations
        #[ec1_1, ec2_1, ec3_1, _, _, _, _, _] = self.encoder(reconstruction)
        [c1_1e1, c1_2e1, c2_1e1, c2_2e1, c3_1e1, c3_2e1, c3_3e1, c4_1e1, c4_2e1, c4_3e1,
         c5_1e1, c5_2e1, c5_3e1, _, _, _, _, _] = self.encoder(reconstruction)

        # Latent Space Classifier
        if self.classifier is not None:
            y_p = self.classifier(z)

        # Reconstruction loss is BCE
        reconstruction_loss = get_reconstruction_error(x, reconstruction)
        weighted_reconstruction_loss = reconstruction_loss * self.content_weight

        # Latent Space Loss us Kullback-Leibler Divergence
        #kl_loss = 1 + z_log_var_0 - tf.square(z_mean_0) - tf.exp(z_log_var_0)
        #kl_loss = tf.reduce_mean(kl_loss)
        kl_loss = get_kl_loss(z_mean, z_log_var)
        weighted_kl_loss = kl_loss * self.kl_weight

        # Style loss is gram matrix comparison of matched encoder and decoder layers for the
        # original input image and the reconstructed image

        # Encoded vs Decoded Img Activations in the Encoder
        # sl1 = get_style_loss(ec1_0, ec1_1, self.img_dim, self.img_dim)
        # sl2 = get_style_loss(ec2_0, ec2_1, self.img_dim, self.img_dim)
        # sl3 = get_style_loss(ec3_0, ec3_1, self.img_dim, self.img_dim)

        sl1 = get_style_loss(c1_1e0, c1_1e1, self.img_dim, self.img_dim)
        sl2 = get_style_loss(c1_2e0, c1_2e1, self.img_dim, self.img_dim)
        sl3 = get_style_loss(c2_1e0, c2_1e1, self.img_dim, self.img_dim)
        sl4 = get_style_loss(c2_2e0, c2_2e1, self.img_dim, self.img_dim)
        sl5 = get_style_loss(c3_1e0, c3_1e1, self.img_dim, self.img_dim)
        sl6 = get_style_loss(c3_2e0, c3_2e1, self.img_dim, self.img_dim)
        sl7 = get_style_loss(c3_3e0, c3_3e1, self.img_dim, self.img_dim)
        sl8 = get_style_loss(c4_1e0, c4_1e1, self.img_dim, self.img_dim)
        sl9 = get_style_loss(c4_2e0, c4_2e1, self.img_dim, self.img_dim)
        sl10 = get_style_loss(c4_3e0, c4_3e1, self.img_dim, self.img_dim)
        sl11 = get_style_loss(c5_1e0, c5_1e1, self.img_dim, self.img_dim)
        sl12 = get_style_loss(c5_2e0, c5_2e1, self.img_dim, self.img_dim)
        sl13 = get_style_loss(c5_3e0, c5_3e1, self.img_dim, self.img_dim)

        # Encoder vs Decoder Activations for the original image
        # sl4 = get_style_loss(ec1_0, dc2, self.img_dim, self.img_dim)
        # sl5 = get_style_loss(ec2_0, dc3, self.img_dim, self.img_dim)
        # sl6 = get_style_loss(ec3_0, reshape, self.img_dim, self.img_dim)

        sl14 = get_style_loss(c5_3e0, c5_3d, self.img_dim, self.img_dim)
        sl15 = get_style_loss(c5_2e0, c5_2d, self.img_dim, self.img_dim)
        sl16 = get_style_loss(c5_1e0, c5_1d, self.img_dim, self.img_dim)

        sl17 = get_style_loss(c4_3e0, c4_3d, self.img_dim, self.img_dim)
        sl18 = get_style_loss(c4_2e0, c4_2d, self.img_dim, self.img_dim)
        sl19 = get_style_loss(c4_1e0, c4_1d, self.img_dim, self.img_dim)

        sl20 = get_style_loss(c3_3e0, c3_3d, self.img_dim, self.img_dim)
        sl21 = get_style_loss(c3_2e0, c3_2d, self.img_dim, self.img_dim)
        sl22 = get_style_loss(c3_1e0, c3_1d, self.img_dim, self.img_dim)

        sl23 = get_style_loss(c2_2e0, c2_2d, self.img_dim, self.img_dim)
        sl24 = get_style_loss(c2_1e0, c2_1d, self.img_dim, self.img_dim)

        sl25 = get_style_loss(c1_2e0, c1_2d, self.img_dim, self.img_dim)
        #sl26 = get_style_loss(c1_1e0, reconstruction, self.img_dim, self.img_dim)

        style_loss = tf.reduce_mean((sl1 + sl2 + sl3 + sl4 + sl5 + sl6
                                     + sl7 + sl8 + sl9 + sl10 + sl11 + sl12 + sl13
                                     + sl14 + sl15 + sl16 + sl17 + sl18 + sl19 + sl20
                                     + sl21 + sl22 + sl23 + sl24 + sl25) / 25.)

        weighted_style_loss = style_loss * self.style_weight

        # Latent space classifier loss
        weighted_cce = 0.0
        if self.classifier is not None:
            class_cce = keras.losses.CategoricalCrossentropy()
            categorical_cross_entropy = class_cce(y, y_p)
            weighted_cce = self.class_weight * categorical_cross_entropy

            # Latent space accuracy
            self.class_accuracy.update_state(tf.math.argmax(y, axis=1), tf.math.argmax(y_p, axis=1))
            accuracy = self.class_accuracy.result()

        total_loss = weighted_reconstruction_loss + weighted_kl_loss + weighted_style_loss + weighted_cce

        metrics = {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "style_loss": style_loss,
        }

        if self.classifier is not None:
            metrics["accuracy"] = accuracy
            metrics["categorical_cross_entropy"] = categorical_cross_entropy

        return metrics


    def train_step(self, data):
        """
        Training step for TensorFlow model updates

        :param data: Input data (x,y) of images and labels
        :type data: tuple
        """
        #if isinstance(data, tuple):
        #    data = data[0]
        assert(isinstance(data, tuple))
        x = data[0]
        y = data[1]

        with tf.GradientTape() as tape:
            loss = self.get_cost(x, y)

        grads = tape.gradient(loss['loss'], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss


    def test_step(self, data):
        """
        Testing step for TensorFlow model updates

        :param data: Input data (x,y) of images and labels
        :type data: tuple
        """
        ##if isinstance(data, tuple):
        ##    data = data[0]
        assert(isinstance(data, tuple))
        x = data[0]
        y = data[1]

        return self.get_cost(x, y)
        