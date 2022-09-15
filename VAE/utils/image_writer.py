#!/usr/bin/env python3

import io
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

class ImageWriterCallback(keras.callbacks.Callback):
    def __init__(self, file_writer, model, imgs):
        super(ImageWriterCallback, self).__init__()
        self.file_writer = file_writer
        self.model = model
        self.imgs = imgs

    def on_epoch_end(self, epoch, logs=None):

        N = 3 # number of samples to grab
        encoder_outputs = self.model.encoder(self.imgs[:N])
        z_0 = encoder_outputs[-1]
        decoder_outputs = self.model.decoder(z_0)
        rec = decoder_outputs[-1]

        img = self.imgs[:N]
        descaled_rec = rec

        fig, axs = plt.subplots(2, N)

        axs[0][0].set_title('Original Image')
        axs[1][0].set_title('Reconstruction')


        for i in range(N):
            axs[0][i].imshow(img[i])
            axs[0][i].xticklabels = ([])

            axs[1][i].imshow(descaled_rec[i])
            axs[1][i].xticklabels = ([])

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        output_image = tf.image.decode_png(buffer.getvalue(), channels = 4)
        output_image = tf.expand_dims(output_image, 0)

        with self.file_writer.as_default():
            tf.summary.image("Model Predictions", output_image, step=epoch)
