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

"""
## Setup
"""
import numpy as np
from datetime import datetime as dt
import tensorflow as tf
from tensorflow import keras
from VAE.utils.VAE_utils import RGB_Dataset
from VAE.include import VAE_arch
from VAE import VAE

from sklearn.preprocessing import LabelBinarizer

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
    OUTPUT_PREPROCESS_PATH = args.outputPreprocessPath

    EARLY_STOPPING_FLAG = args.earlyStopping
    EARLY_STOPPING_PATIENCE = args.earlyStoppingPatience

    WEIGHT_SL = args.styleLoss
    WEIGHT_KL = args.klLoss
    WEIGHT_BCE = IMG_DIM*IMG_DIM
    WEIGHT_CLASSIFIER = args.classLoss

    ENABLE_CLASSIFIER_FLAG = args.enable_classifier

    if LOG_DIR==None:
        LOG_DIR=os.path.abspath(os.curdir)

    print(f"LOG_DIR: {LOG_DIR}")

    PATH_TO_ENCODER = args.encoderPath
    PATH_TO_DECODER = args.decoderPath

    """
    ## Build the encoder
    """

    if PATH_TO_ENCODER:
        encoder = keras.models.load_model(os.path.join(PATH_TO_ENCODER, "encoder"))
    else:
        encoder = VAE_arch.encoder_3conv7c(IMG_DIM, IMG_CH, LATENT_DIM)
        #encoder = VAE_arch.encoder_3conv7d(IMG_DIM, IMG_CH, LATENT_DIM)

    encoder.summary()

    """
    ## Build the decoder
    """

    if PATH_TO_DECODER:
        decoder = keras.models.load_model(os.path.join(PATH_TO_DECODER,"decoder"))
    else:
        # Decoding network
        decoder = VAE_arch.decoder_3conv7c(LATENT_DIM)

    decoder.summary()

    """
    ## Import Training Data
    """
    dataset = RGB_Dataset()

    orig_imgs, labels, orig_list = dataset.load_rgb(
        IMG_DIM, 
        IMG_CH, 
        groundTruthFile=GT_FILE, 
        output_preprocess_path=OUTPUT_PREPROCESS_PATH
    )

    # randomize order
    imgs = np.random.permutation(orig_imgs)

    # split into training and validation batches
    num_imgs = imgs.shape[0]
    num_valImgs = np.math.floor(num_imgs * VALIDATION_SPLIT)
    val_steps = np.math.floor(num_valImgs / MINI_BATCH)

    # normalize training data
    normalised_input = (imgs) / np.max(imgs)
    imgs = normalised_input
    print(f'Images Normalized: {np.max(imgs)}')

    """
    ## Setup the Classifier
    """
    # By default, classifier is 'None' to disable
    classifier = None
    # Class labels are extracted from input data
    class_labels = [label[0] for label in labels]
    # Class labels put into ones-hot vector
    labels_ones_hot = LabelBinarizer().fit_transform(class_labels)
    # If enabled, build the classifier
    if ENABLE_CLASSIFIER_FLAG:
        classifier = VAE_arch.latent_classifier_arch(LATENT_DIM, len(set(class_labels)))

    """
    ## Train the VAE
    """
    if (PATH_TO_DECODER == PATH_TO_ENCODER) and (PATH_TO_DECODER != None):
        tb_path = PATH_TO_DECODER

    else:
        now = "{:%Y%m%dT%H%M}".format(dt.now())
        tb_path = os.path.join(LOG_DIR, now)

    # VAE Instantiation
    vae = VAE(encoder, decoder, classifier, WEIGHT_BCE, WEIGHT_SL, WEIGHT_KL, WEIGHT_CLASSIFIER, IMG_DIM)

    # Model Compile
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    # Callbacks

    # Tensorboard
    callbacks = [tf.keras.callbacks.TensorBoard(
        log_dir=(tb_path), 
        histogram_freq=10,
    )]

    # EarlyStopping
    if EARLY_STOPPING_FLAG:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1E-3,
            patience=EARLY_STOPPING_PATIENCE,
            verbose=True,
            mode='auto',
            restore_best_weights=False
        ))

    # Allow early-terminate by CTRL+C
    try:
        # Execute Fitting
        vae.fit(
            #imgs, 
            x=imgs,
            y=labels_ones_hot,
            epochs=EPOCHS, 
            batch_size=MINI_BATCH, 
            callbacks=callbacks,
            use_multiprocessing=True, 
            validation_split=VALIDATION_SPLIT, 
            validation_steps=val_steps,
            initial_epoch=INITIAL_EPOCH
        )
    except KeyboardInterrupt:
        print('Terminating training')

    # Save network for future use
    encoder.save(filepath=(os.path.join(tb_path, "encoder")))
    decoder.save(filepath=(os.path.join(tb_path, "decoder")))
    if classifier is not None:
        classifier.save(filepath=(os.path.join(tb_path, "classifier")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, default = '.')
    parser.add_argument('--encoderPath', type=str, default=None)
    parser.add_argument('--decoderPath', type=str, default=None)
    parser.add_argument('--logDir', type=str, default=os.path.join('logs', 'fit'))
    parser.add_argument('--imgDim', type=int, default=128)
    parser.add_argument('--imgCh', type=int, default=3)
    parser.add_argument('--miniBatch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--latentDim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--valSplit', type=float, default=0.15)
    parser.add_argument('--epochStart', type=int, default=0)
    parser.add_argument('--outputPreprocessPath', type=str, default=None)
    parser.add_argument('--earlyStopping', action='store_true')
    parser.add_argument('--earlyStoppingPatience', type=int, default=8)
    parser.add_argument('--styleLoss', type=float, default=1e7)
    parser.add_argument('--klLoss', type=float, default=0.5)
    parser.add_argument('--classLoss', type=float, default=1.0)
    parser.add_argument('--enable-classifier', action='store_true', help='Enables the classifier on the latent space')

    args = parser.parse_args()
    print(args)

    main()
