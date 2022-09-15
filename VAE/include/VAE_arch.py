#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.layers import SpectralNormalization

"""
## Create a sampling layer



"""

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


"""
## Define various encoder architectures
"""

def encoder_2conv(IMG_DIM, IMG_CH, LATENT_DIM):

    encoder_inputs = keras.Input(shape=(IMG_DIM, IMG_DIM, IMG_CH))  # if this is 128
    x = layers.Conv2D(32, (3, 3), activation="relu", strides=2, padding="same")(encoder_inputs)  # 64
    x = layers.Conv2D(64, (3, 3), activation="relu", strides=2, padding="same")(x)  # 32
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)  # <- this is the bottleneck
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder

def encoder_2conv7(IMG_DIM, IMG_CH, LATENT_DIM):
    #on color set; this generated D:\Ashley_ML\VAE\VAE\VAE_logs\20201222T1642
    encoder_inputs = keras.Input(shape=(IMG_DIM, IMG_DIM, IMG_CH))  # if this is 128
    x = layers.Conv2D(128, (5, 5), activation="relu", strides=2, padding="same")(encoder_inputs)  # 64
    x = layers.Conv2D(256, (5, 5), activation="relu", strides=2, padding="same")(x)  # 32
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)  # <- this is the bottleneck
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder

def encoder_2conv7a(IMG_DIM, IMG_CH, LATENT_DIM):

    encoder_inputs = keras.Input(shape=(IMG_DIM, IMG_DIM, IMG_CH))  # if this is 128
    x = layers.Conv2D(64, (5, 5), activation="relu", strides=2, padding="same")(encoder_inputs)  # 64
    x = layers.Conv2D(128, (5, 5), activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(256, (5, 5), activation="relu", strides=2, padding="same")(x)# 32
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)  # <- this is the bottleneck
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder

def encoder_2conv7b(IMG_DIM, IMG_CH, LATENT_DIM):

    encoder_inputs = keras.Input(shape=(IMG_DIM, IMG_DIM, IMG_CH))  # if this is 128
    x = layers.Conv2D(128, (5, 5), activation="relu", strides=2, padding="same", name="conv1encoder")(encoder_inputs)  # 64
    x = layers.Conv2D(256, (5, 5), activation="relu", strides=2, padding="same", name="conv2encoder")(x)  # 32
    x = layers.Conv2D(512, (5, 5), activation="relu", strides=2, padding="same", name="conv3encoder")(x)
    x = layers.Flatten(name='flattenEncoder')(x)
    x = layers.Dense(512, activation="relu", name='denseEncoder')(x)  # <- this is the bottleneck
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling(name="sampling")([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder

def encoder_3conv7c(IMG_DIM, IMG_CH, LATENT_DIM):

    encoder_inputs = keras.Input(shape=(IMG_DIM, IMG_DIM, IMG_CH))  # if this is 128
    conv1 = layers.Conv2D(128, (5, 5), activation="relu", strides=2, padding="same", name="conv1encoder")(encoder_inputs)  # 64
    conv2 = layers.Conv2D(256, (5, 5), activation="relu", strides=2, padding="same", name="conv2encoder")(conv1)  # 32
    conv3 = layers.Conv2D(512, (5, 5), activation="relu", strides=2, padding="same", name="conv3encoder")(conv2)
    flat = layers.Flatten(name='flattenEncoder')(conv3)
    dense = layers.Dense(512, activation="relu", name='denseEncoder')(flat)  # <- this is the bottleneck
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(dense)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(dense)
    z = Sampling(name="sampling")([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [conv1, conv2, conv3, flat, dense, z_mean, z_log_var, z], name="encoder")

    return encoder

def encoder_3conv7d(IMG_DIM, IMG_CH, LATENT_DIM):

    encoder_inputs = keras.Input(shape=(IMG_DIM, IMG_DIM, IMG_CH))  # if this is 128
    conv1 = layers.Conv2D(128, (7, 7), activation="relu", strides=2, padding="same", name="conv1encoder")(encoder_inputs)  # 64
    conv2 = layers.Conv2D(256, (7, 7), activation="relu", strides=2, padding="same", name="conv2encoder")(conv1)  # 32
    conv3 = layers.Conv2D(512, (7, 7), activation="relu", strides=2, padding="same", name="conv3encoder")(conv2)
    flat = layers.Flatten(name='flattenEncoder')(conv3)
    dense = layers.Dense(512, activation="relu", name='denseEncoder')(flat)  # <- this is the bottleneck
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(dense)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(dense)
    z = Sampling(name="sampling")([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [conv1, conv2, conv3, flat, dense, z_mean, z_log_var, z], name="encoder")

    return encoder


def encoder_3conv7d_sns(IMG_DIM, IMG_CH, LATENT_DIM):

    encoder_inputs = keras.Input(shape=(IMG_DIM, IMG_DIM, IMG_CH))  # if this is 128

    conv1 = SpectralNormalization(layers.Conv2D(128, (7, 7), activation="relu", strides=2, padding="same", name="conv1encoder"))(encoder_inputs)  # 64
    conv2 = SpectralNormalization(layers.Conv2D(256, (7, 7), activation="relu", strides=2, padding="same", name="conv2encoder"))(conv1) # 32
    conv3 = SpectralNormalization(layers.Conv2D(512, (7, 7), activation="relu", strides=2, padding="same", name="conv3encoder"))(conv2)
    flat = layers.Flatten(name='flattenEncoder')(conv3)
    dense = SpectralNormalization(layers.Dense(512, activation="relu", name='denseEncoder'))(flat)  # <- this is the bottleneck
    z_mean = SpectralNormalization(layers.Dense(LATENT_DIM, name="z_mean"))(dense)
    z_log_var = SpectralNormalization(layers.Dense(LATENT_DIM, name="z_log_var"))(dense)
    z = Sampling(name="sampling")([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [conv1, conv2, conv3, flat, dense, z_mean, z_log_var, z], name="encoder")

    return encoder


def encoder_vgg_sn(IMG_DIM, IMG_CH, LATENT_DIM):

    encoder_inputs = keras.Input(shape=(IMG_DIM, IMG_DIM, IMG_CH))

    conv1_1 = SpectralNormalization(layers.Conv2D(64, (5, 5), activation="relu", strides=1, padding="same"), name="conv1_1en")(encoder_inputs)
    conv1_2 = SpectralNormalization(layers.Conv2D(64, (5, 5), activation="relu", strides=1, padding="same"), name="conv1_2en")(conv1_1)
    mp_1 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same", name="maxpool_1")(conv1_2)

    conv2_1 = SpectralNormalization(layers.Conv2D(128, (5, 5), activation="relu", strides=1, padding="same"), name="conv2_1en")(mp_1)
    conv2_2 = SpectralNormalization(layers.Conv2D(128, (5, 5), activation="relu", strides=1, padding="same"), name="conv2_2en")(conv2_1)
    mp_2 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same", name="maxpool_2")(conv2_2)

    conv3_1 = SpectralNormalization(layers.Conv2D(256, (5, 5), activation="relu", strides=1, padding="same"), name="conv3_1en")(mp_2)
    conv3_2 = SpectralNormalization(layers.Conv2D(256, (5, 5), activation="relu", strides=1, padding="same"), name="conv3_2en")(conv3_1)
    conv3_3 = SpectralNormalization(layers.Conv2D(256, (5, 5), activation="relu", strides=1, padding="same"), name="conv3_3en")(conv3_2)
    mp_3 = layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv3_3)

    conv4_1 = SpectralNormalization(layers.Conv2D(512, (5, 5), activation="relu", strides=1, padding="same"), name="conv4_1en")(mp_3)
    conv4_2 = SpectralNormalization(layers.Conv2D(512, (5, 5), activation="relu", strides=1, padding="same"), name="conv4_2en")(conv4_1)
    conv4_3 = SpectralNormalization(layers.Conv2D(512, (5, 5), activation="relu", strides=1, padding="same"), name="conv4_3en")(conv4_2)
    mp_4 = layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv4_3)

    conv5_1 = SpectralNormalization(layers.Conv2D(512, (5, 5), activation="relu", strides=1, padding="same"), name="conv5_1en")(mp_4)
    conv5_2 = SpectralNormalization(layers.Conv2D(512, (5, 5), activation="relu", strides=1, padding="same"), name="conv5_2en")(conv5_1)
    conv5_3 = SpectralNormalization(layers.Conv2D(512, (5, 5), activation="relu", strides=1, padding="same"), name="conv5_3en")(conv5_2)
    mp_5 = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(conv5_3)

    flat = layers.Flatten(name="flattenEncoder")(mp_5)
    dense = SpectralNormalization(layers.Dense(512, activation="relu", name="denseEncoder"))(flat)
    z_mean = SpectralNormalization(layers.Dense(LATENT_DIM, activation="relu", name="z_mean"))(dense)
    z_log_var = SpectralNormalization(layers.Dense(LATENT_DIM, name="z_log_var"))(dense)
    z = Sampling(name="sampling")([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs,
                          [conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv4_1, conv4_2, conv4_3,
                           conv5_1, conv5_2, conv5_3, flat, dense, z_mean, z_log_var, z], name="encoder")

    return encoder



def encoder_3conv(IMG_DIM, IMG_CH, LATENT_DIM):
    encoder_inputs = keras.Input(shape=(IMG_DIM, IMG_DIM, IMG_CH))  # if this is 128
    x = layers.Conv2D(32, (3, 3), activation="relu", strides=2, padding="same")(encoder_inputs)  # 64
    x = layers.Conv2D(64, (3, 3), activation="relu", strides=2, padding="same")(x)  # 32
    x = layers.Conv2D(128, (3, 3), activation="relu", strides=2, padding="same")(x) # 16
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)  # <- this is the bottleneck
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder

def encoder_4conv(IMG_DIM, IMG_CH, LATENT_DIM):
    encoder_inputs = keras.Input(shape=(IMG_DIM, IMG_DIM, IMG_CH))  # if this is 128
    x = layers.Conv2D(32, (3, 3), activation="relu", strides=2, padding="same")(encoder_inputs)  # 64
    x = layers.Conv2D(64, (3, 3), activation="relu", strides=2, padding="same")(x)  # 32
    x = layers.Conv2D(128, (3, 3), activation="relu", strides=2, padding="same")(x) # 16
    x = layers.Conv2D(256, (3, 3), activation="relu", strides=2, padding="same")(x) # 8
    x = layers.Flatten()(x)
    x = layers.Dense(8, activation="relu")(x)  # <- this is the bottleneck
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder

"""
## Define various decoder architectures
"""

def decoder_2conv(LATENT_DIM):
    # Decoding network
    latent_inputs = keras.Input(shape=(LATENT_DIM,))  # "input_2"
    x = layers.Dense(32 * 32 * 256, activation="relu")(latent_inputs)  # "dense_1"
    x = layers.Reshape((32, 32, 256))(x)  # "reshape"
    # x = layers.Conv2DTranspose(256, (3, 3), activation="relu", strides=2, padding="same")(x) # 8
    # x = layers.Conv2DTranspose(128, (3, 3), activation="relu", strides=2, padding="same")(x) #  "conv2d_transpose"
    x = layers.Conv2DTranspose(256, (3, 3), activation="relu", strides=2, padding="same")(x)  # "conv2d_transpose_1"
    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", strides=2, padding="same")(x)  # "conv2d_transpose_2"
    decoder_outputs = layers.Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same")(x)  # "conv2d_transpose_3"
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder

def decoder_2conv7a(LATENT_DIM):
    # Decoding network
    latent_inputs = keras.Input(shape=(LATENT_DIM,))  # "input_2"
    x = layers.Dense(16 * 16 * 512, activation="relu")(latent_inputs)  # "dense_1"
    x = layers.Reshape((16, 16, 512))(x)  # "reshape"
    x = layers.Conv2DTranspose(256, (5, 5), activation="relu", strides=2, padding="same")(x)  # "conv2d_transpose_1"
    x = layers.Conv2DTranspose(128, (5, 5), activation="relu", strides=2, padding="same")(x)  # "conv2d_transpose_2"
    decoder_outputs = layers.Conv2DTranspose(3, (5, 5), strides=2, activation="sigmoid", padding="same")(x)  # "conv2d_transpose_3"
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder

def decoder_3conv7c(LATENT_DIM):
    # Decoding network
    latent_inputs = keras.Input(shape=(LATENT_DIM,))  # "input_2"
    dense = layers.Dense(16 * 16 * 512, activation="relu")(latent_inputs)  # "dense_1"
    reshape = layers.Reshape((16, 16, 512))(dense)  # "reshape"
    conv3 = layers.Conv2DTranspose(256, (5, 5), activation="relu", strides=2, padding="same", name="conv3decoder")(reshape)  # "conv2d_transpose_1"
    conv2 = layers.Conv2DTranspose(128, (5, 5), activation="relu", strides=2, padding="same", name="conv2decoder")(conv3)  # "conv2d_transpose_2"
    decoder_outputs = layers.Conv2DTranspose(3, (5, 5), strides=2, activation="sigmoid", padding="same",name="conv1decoder")(conv2)  # "conv2d_transpose_3"
    decoder = keras.Model(latent_inputs, [dense, reshape, conv3, conv2, decoder_outputs], name="decoder")

    return decoder

def decoder_3conv7d(LATENT_DIM):
    # Decoding network
    latent_inputs = keras.Input(shape=(LATENT_DIM,))  # "input_2"
    dense = layers.Dense(16 * 16 * 512, activation="relu")(latent_inputs)  # "dense_1"
    reshape = layers.Reshape((16, 16, 512))(dense)  # "reshape"
    conv3 = layers.Conv2DTranspose(256, (7, 7), activation="relu", strides=2, padding="same", name="conv3decoder")(reshape)  # "conv2d_transpose_1"
    conv2 = layers.Conv2DTranspose(128, (7, 7), activation="relu", strides=2, padding="same", name="conv2decoder")(conv3)  # "conv2d_transpose_2"
    decoder_outputs = layers.Conv2DTranspose(3, (7, 7), strides=2, activation="sigmoid", padding="same",name="conv1decoder")(conv2)  # "conv2d_transpose_3"
    decoder = keras.Model(latent_inputs, [dense, reshape, conv3, conv2, decoder_outputs], name="decoder")

    return decoder

def decoder_3conv7d_sns(LATENT_DIM):
    # Decoding network
    latent_inputs = keras.Input(shape=(LATENT_DIM,))  # "input_2"
    dense = SpectralNormalization(layers.Dense(16 * 16 * 512, activation="relu"))(latent_inputs)  # "dense_1"
    reshape = layers.Reshape((16, 16, 512))(dense)  # "reshape"
    conv3 = SpectralNormalization(layers.Conv2DTranspose(256, (7, 7), activation="relu", strides=2, padding="same",
                                                         name="conv3decoder"))(reshape)  # "conv2d_transpose_1"
    conv2 = SpectralNormalization(layers.Conv2DTranspose(128, (7, 7), activation="relu", strides=2, padding="same",
                                                         name="conv2decoder"))(conv3)  # "conv2d_transpose_2"
    decoder_outputs = SpectralNormalization(layers.Conv2DTranspose(3, (7, 7), strides=2, activation="sigmoid", padding="same",
                                                                   name="conv1decoder"))(conv2)  # "conv2d_transpose_3"
    decoder = keras.Model(latent_inputs, [dense, reshape, conv3, conv2, decoder_outputs], name="decoder")

    return decoder


def decoder_vgg_sn(IMG_DIM, LATENT_DIM):

    latent_inputs = keras.Input(shape=(LATENT_DIM, ))
    dense = SpectralNormalization(layers.Dense(512*4*4, activation="relu"), name="denseDecoder")(latent_inputs)
    reshape = layers.Reshape((4, 4, 512))(dense)

    conv5_3 = SpectralNormalization(layers.Conv2DTranspose(512, (5, 5), activation="relu", strides=2, padding="same"),
                                    name="conv5_3de")(reshape)
    conv5_2 = SpectralNormalization(layers.Conv2DTranspose(512, (5, 5), activation="relu", strides=1, padding="same"),
                                    name="conv5_2de")(conv5_3)
    conv5_1 = SpectralNormalization(layers.Conv2DTranspose(512, (5, 5), activation="relu", strides=1, padding="same"),
                                    name="conv5_1de")(conv5_2)

    conv4_3 = SpectralNormalization(layers.Conv2DTranspose(512, (5, 5), activation="relu", strides=2, padding="same"),
                                    name="conv4_3de")(conv5_1)
    conv4_2 = SpectralNormalization(layers.Conv2DTranspose(512, (5, 5), activation="relu", strides=1, padding="same"),
                                    name="conv4_2de")(conv4_3)
    conv4_1 = SpectralNormalization(layers.Conv2DTranspose(512, (5, 5), activation="relu", strides=1, padding="same"),
                                    name="conv4_1de")(conv4_2)

    conv3_3 = SpectralNormalization(layers.Conv2DTranspose(256, (5, 5), activation="relu", strides=2, padding="same"),
                                    name="conv3_3de")(conv4_1)
    conv3_2 = SpectralNormalization(layers.Conv2DTranspose(256, (5, 5), activation="relu", strides=1, padding="same"),
                                    name="conv3_2de")(conv3_3)
    conv3_1 = SpectralNormalization(layers.Conv2DTranspose(128, (5, 5), activation="relu", strides=1, padding="same"),
                                    name="conv3_1de")(conv3_2)

    conv2_2 = SpectralNormalization(layers.Conv2DTranspose(128, (5, 5), activation="relu", strides=2, padding="same"),
                                    name="conv2_2de")(conv3_1)
    conv2_1 = SpectralNormalization(layers.Conv2DTranspose(64, (5, 5), activation="relu", strides=1, padding="same"),
                                    name="conv2_1de")(conv2_2)

    conv1_2 = SpectralNormalization(layers.Conv2DTranspose(64, (5, 5), activation="relu", strides=2, padding="same"),
                                    name="conv1_2de")(conv2_1)
    decoder_outputs = SpectralNormalization(layers.Conv2DTranspose(3, (5, 5), activation="relu", strides=1, padding="same"),
                                            name="conv1_1de")(conv1_2)

    decoder = keras.Model(latent_inputs, [dense, reshape, conv5_3, conv5_2, conv5_1, conv4_3, conv4_2, conv4_1, conv3_3, conv3_2,
                                          conv3_1, conv2_2, conv2_1, conv1_2, decoder_outputs], name="decoder")

    return decoder


def decoder_3conv(LATENT_DIM):
    # DONT CALL THIS ONE: IT ISN"T DONE

    # Decoding network
    latent_inputs = keras.Input(shape=(LATENT_DIM,))  # "input_2"
    x = layers.Dense(16 * 16 * 128, activation="relu")(latent_inputs)  # "dense_1"
    x = layers.Reshape((16, 16, 128))(x)  # "reshape"
    # x = layers.Conv2DTranspose(256, (3, 3), activation="relu", strides=2, padding="same")(x) # 8
    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", strides=2, padding="same")(x) #  "conv2d_transpose"
    x = layers.Conv2DTranspose(256, (3, 3), activation="relu", strides=2, padding="same")(x)  # "conv2d_transpose_1"
    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", strides=2, padding="same")(x)  # "conv2d_transpose_2"
    decoder_outputs = layers.Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same")(x)  # "conv2d_transpose_3"
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder

def decoder_4conv(LATENT_DIM):

    # DONT CALL THIS ONE: IT ISN"T DONE

    # Decoding network
    latent_inputs = keras.Input(shape=(LATENT_DIM,))  # "input_2"
    x = layers.Dense(32 * 32 * 128, activation="relu")(latent_inputs)  # "dense_1"
    x = layers.Reshape((32, 32, 128))(x)  # "reshape"
    x = layers.Conv2DTranspose(256, (3, 3), activation="relu", strides=2, padding="same")(x) # 8
    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", strides=2, padding="same")(x) #  "conv2d_transpose"
    x = layers.Conv2DTranspose(256, (3, 3), activation="relu", strides=2, padding="same")(x)  # "conv2d_transpose_1"
    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", strides=2, padding="same")(x)  # "conv2d_transpose_2"
    decoder_outputs = layers.Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same")(x)  # "conv2d_transpose_3"
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder

def betaVAE_encoder(IMG_DIM, IMG_CH, LATENT_DIM):

    encoder_inputs = keras.Input(shape=(IMG_DIM, IMG_DIM, IMG_CH))  # if this is 128
    conv1 = layers.Conv2D(32, (4, 4), activation="relu", strides=2, padding="same", name="conv1encoder")(encoder_inputs)
    conv2 = layers.Conv2D(32, (4, 4), activation="relu", strides=2, padding="same", name="conv2encoder")(conv1)
    conv3 = layers.Conv2D(32, (4, 4), activation="relu", strides=2, padding="same", name="conv3encoder")(conv2)
    conv4 = layers.Conv2D(32, (4, 4), activation="relu", strides=2, padding="same", name="conv4encoder")(conv3)
    conv5 = layers.Conv2D(32, (4, 4), activation="relu", strides=2, padding="same", name="conv5encoder")(conv4)
    flat = layers.Flatten(name='flattenEncoder')(conv5)
    fcc_1 = layers.Dense(512, activation="relu", name="fcc_1")(flat)
    fcc_2 = layers.Dense(512, activation = "relu", name="fcc_2")(fcc_1)
    #dense = layers.Dense(512, activation="relu", name='denseEncoder')(flat)  # <- this is the bottleneck
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(fcc_2)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(fcc_2)
    z = Sampling(name="sampling")([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [conv1, conv2, conv3, conv4, conv5, fcc_1, fcc_2, z_mean, z_log_var, z], name="encoder")

    return encoder

def betaVAE_decoder(LATENT_DIM):

    # Decoding network
    latent_inputs = keras.Input(shape=(LATENT_DIM,))  # "input_2"
    fcc_3 = layers.Dense(512, activation="relu", name="fcc_3")(latent_inputs)
    fcc_4 = layers.Dense(512, activation = "relu", name="fcc_4")(fcc_3)
    reshape = layers.Reshape((4, 4, 32))(fcc_4)
    conv5 = layers.Conv2DTranspose(32, (4, 4), activation="relu", strides=2, padding="same", name="conv5decoder")(reshape)
    conv4 = layers.Conv2DTranspose(32, (4, 4), activation="relu", strides=2, padding="same", name="conv4decoder")(conv5)
    conv3 = layers.Conv2DTranspose(32, (4, 4), activation="relu", strides=2, padding="same", name="conv3decoder")(conv4)
    conv2 = layers.Conv2DTranspose(32, (4, 4), activation="relu", strides=2, padding="same", name="conv2decoder")(conv3)
    decoder_outputs = layers.Conv2DTranspose(3, (4, 4), strides=2, padding="same", activation="sigmoid", name="conv1decoder")(conv2)  # "conv2d_transpose_3"
    decoder = keras.Model(latent_inputs, [fcc_3, fcc_4, reshape, conv5, conv4, conv3, conv2, decoder_outputs], name="decoder")
    return decoder


"""
## Classifier Architecture
"""

def latent_classifier_arch(latent_dim, total_classes):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(latent_dim,), name='input'),
        layers.Dense(70, activation='relu', name='dense_0'),
        layers.Dense(30, activation='relu', name='dense_1'),
        layers.Dense(20, activation='relu', name='dense_2'),
        layers.Dense(total_classes, activation='softmax', name='output'),
    ])
    #model.build((None, latent_dim))

    #model.build()
    #model.summary()
    #exit()

    return model


"""
## Version of VAE with encoder and decoder defined together
"""

def totalVAE(IMG_DIM, IMG_CH, LATENT_DIM):
    encoder_inputs = keras.Input(shape=(IMG_DIM, IMG_DIM, IMG_CH))  # if this is 128
    econv1= layers.Conv2D(128, (5, 5), activation="relu", strides=2, padding="same", name="conv1encoder")(
        encoder_inputs)  # 64
    econv2 = layers.Conv2D(256, (5, 5), activation="relu", strides=2, padding="same", name="conv2encoder")(econv1)  # 32
    econv3 = layers.Conv2D(512, (5, 5), activation="relu", strides=2, padding="same", name="conv3encoder")(econv2)
    eflat = layers.Flatten(name='flattenEncoder')(econv3)
    edense = layers.Dense(512, activation="relu", name='denseEncoder')(eflat)  # <- this is the bottleneck

    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(edense)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(edense)
    z = Sampling(name="sampling")([z_mean, z_log_var])

   # decoder
    latent_inputs = layers.Reshape((-1, LATENT_DIM))(z)
    ddense = layers.Dense(16 * 16 * 512, activation="relu")(latent_inputs)  # "dense_1"
    dreshape = layers.Reshape((16, 16, 512))(ddense)  # "reshape"
    dconv3 = layers.Conv2DTranspose(256, (5, 5), activation="relu", strides=2, padding="same",name="conv3decoder")(dreshape)  # "conv2d_transpose_1"
    dconv2 = layers.Conv2DTranspose(128, (5, 5), activation="relu", strides=2, padding="same", name="conv2decoder")(dconv3)  # "conv2d_transpose_2"
    decoder_outputs = layers.Conv2DTranspose(3, (5, 5), strides=2, activation="sigmoid", padding="same",name="conv1decoder")(
        dconv2)  # "conv2d_transpose_3"

    vae = keras.Model(encoder_inputs, [econv1, econv2, econv3, eflat, edense, z_mean, z_log_var, z,
                                       latent_inputs, ddense, dreshape, dconv3, dconv2, decoder_outputs], name="totalVAE")
    return vae