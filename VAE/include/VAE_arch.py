#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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