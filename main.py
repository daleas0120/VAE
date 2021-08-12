#!/usr/bin/env python

#"""
#Title: Variational AutoEncoder
#Improvised from: [fchollet](https://twitter.com/fchollet)
#Date created: 2020/05/03
#Last modified: 2021/07/01
#https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py

#TO LAUNCH TENSORBOARD : python -m tensorboard.main --logdir=PATH\TO\LOG_DIR

#"""

import argparse
import sys
import os
cwd = os.path.abspath(os.path.dirname( __file__ ))
sys.path.append(cwd)
print('CWD: '+cwd)

"""
## Setup
"""
import numpy as np
from datetime import datetime as dt
import tensorflow as tf
from tensorflow import keras
from utils.VAE_utils import RGB_Dataset
from utils.VAE_utils import style_loss
from include import VAE_arch

print('Keras: '+keras.__version__)
print('Tensorflow: ' + tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def main():

    IMG_DIM = args.imgDim
    IMG_CH = args.imgCh
    MINI_BATCH = args.miniBatch
    EPOCHS = args.epochs
    INITIAL_EPOCH = args.epochStart
    LATENT_DIM = args.latentDim
    LEARNING_RATE = args.lr
    VALIDATION_SPLIT = args.valSplit
    GT_FILE = args.labels
    LOG_DIR = args.logDir

    WEIGHT_SL = 1e7
    WEIGHT_KL = -0.5
    WEIGHT_BCE = IMG_DIM*IMG_DIM

    if LOG_DIR==None:
        LOG_DIR=os.path.abspath(os.curdir)

    print("LOG_DIR: " + LOG_DIR)

    PATH_TO_ENCODER = args.encoderPath
    PATH_TO_DECODER = args.decoderPath

    """
    ## Build the encoder
    """

    if PATH_TO_ENCODER:
        encoder = keras.models.load_model(PATH_TO_ENCODER+"/encoder")
    else:
        encoder = VAE_arch.encoder_3conv7c(IMG_DIM, IMG_CH, LATENT_DIM)

    encoder.summary()

    """
    ## Build the decoder
    """

    if PATH_TO_DECODER:
        decoder = keras.models.load_model(PATH_TO_DECODER+"/decoder")
    else:
        # Decoding network
        decoder = VAE_arch.decoder_3conv7c(LATENT_DIM)

    decoder.summary()

    """
    ## Define the VAE as a `Model` with a custom `train_step`
    """

    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder

        def train_step(self, data):
            if isinstance(data, tuple):
                data = data[0]
            with tf.GradientTape() as tape:
                content_weight = WEIGHT_BCE
                style_weight = WEIGHT_SL
                kl_weight = WEIGHT_KL

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
                reconstruction_loss *= content_weight

                # Latent Space Loss us Kullback-Leibler Divergence
                kl_loss = 1 + z_log_var_0 - tf.square(z_mean_0) - tf.exp(z_log_var_0)
                kl_loss = tf.reduce_mean(kl_loss)
                kl_loss *= kl_weight

                # Style loss is gram matrix comparison of matched encoder and decoder layers for the
                # original input image and the reconstructed image

                # Encoded vs Decoded Img Activations in the Encoder
                sl1 = style_loss(ec1_0, ec1_1, IMG_DIM, IMG_DIM)
                sl2 = style_loss(ec2_0, ec2_1, IMG_DIM, IMG_DIM)
                sl3 = style_loss(ec3_0, ec3_1, IMG_DIM, IMG_DIM)

                # Encoder vs Decoder Activations for the original image
                sl4 = style_loss(ec1_0, dc2, IMG_DIM, IMG_DIM)
                sl5 = style_loss(ec2_0, dc3, IMG_DIM, IMG_DIM)
                sl6 = style_loss(ec3_0, reshape, IMG_DIM, IMG_DIM)

                sL = tf.reduce_mean((style_weight / 6) * (sl1 + sl2 + sl3 + sl4 + sl5 + sl6))

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

            content_weight = WEIGHT_BCE
            style_weight = WEIGHT_SL
            kl_weight = WEIGHT_KL

            [ec1_0, ec2_0, ec3_0, _, _, z_mean_0, z_log_var_0, z_0] = self.encoder(data)

            [_, reshape, dc3, dc2, reconstruction] = self.decoder(z_0)

            [ec1_1, ec2_1, ec3_1, _, _, _, _, _] = self.encoder(reconstruction)

            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= content_weight

            kl_loss = 1 + z_log_var_0 - tf.square(z_mean_0) - tf.exp(z_log_var_0)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= kl_weight

            # Get style losses

            # Encoded vs Decoded Img Activations in the Encoder
            sl1 = style_loss(ec1_0, ec1_1, IMG_DIM, IMG_DIM)
            sl2 = style_loss(ec2_0, ec2_1, IMG_DIM, IMG_DIM)
            sl3 = style_loss(ec3_0, ec3_1, IMG_DIM, IMG_DIM)

            # Encoder vs Decoder Activations
            sl4 = style_loss(ec1_0, dc2, IMG_DIM, IMG_DIM)
            sl5 = style_loss(ec2_0, dc3, IMG_DIM, IMG_DIM)
            sl6 = style_loss(ec3_0, reshape, IMG_DIM, IMG_DIM)

            sL = tf.reduce_mean((style_weight / 6) * (sl1 + sl2 + sl3 + sl4 + sl5 + sl6))

            total_loss = reconstruction_loss + kl_loss + sL


            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "style_loss": sL,
            }

    """
    ## Import Training Data
    """
    dataset = RGB_Dataset()

    orig_imgs, labels, orig_list = dataset.load_rgb(IMG_DIM, IMG_CH, groundTruthFile=GT_FILE)

    # randomize order
    imgs = np.random.permutation(orig_imgs)

    # split into training and validation batches
    num_imgs = imgs.shape[0]
    num_valImgs = np.math.floor(num_imgs * VALIDATION_SPLIT)
    val_steps = np.math.floor(num_valImgs / MINI_BATCH)

    # normalize training data
    normalised_input = (imgs) / np.max(imgs)
    imgs = normalised_input
    print('Images Normalized: '+str(np.max(imgs)))

    """
    ## Train the VAE
    """
    if (PATH_TO_DECODER == PATH_TO_ENCODER) and (PATH_TO_DECODER != None):
        tb_path = PATH_TO_DECODER

    else:
        now = "{:%Y%m%dT%H%M}".format(dt.now())
        tb_path = LOG_DIR+"/"+now

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=(tb_path), histogram_freq=10)

    vae.fit(imgs, epochs=EPOCHS, batch_size=MINI_BATCH, callbacks=[tensorboard_callback],
            use_multiprocessing=True, validation_split=VALIDATION_SPLIT, validation_steps=val_steps,
            initial_epoch=INITIAL_EPOCH)

    # Save network for future use
    encoder.save(filepath=(tb_path + "/encoder"))
    decoder.save(filepath=(tb_path + "/decoder"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, default = '.')
    parser.add_argument('--encoderPath', type=str, default=None)
    parser.add_argument('--decoderPath', type=str, default=None)
    parser.add_argument('--logDir', type=str, default=None)
    parser.add_argument('--imgDim', type=int, default=128)
    parser.add_argument('--imgCh', type=int, default=3)
    parser.add_argument('--miniBatch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--latentDim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--valSplit', type=float, default=0.15)
    parser.add_argument('--epochStart', type=int, default=0)

    args = parser.parse_args()
    print(args)

    main()
