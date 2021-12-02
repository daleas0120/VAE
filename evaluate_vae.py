#!/usr/bin/env python

"""
## Setup
"""

import argparse
from json import decoder
import cv2
import os
from datetime import datetime
import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from VAE.utils import writeResults
from VAE.utils.VAE_utils import RGB_Dataset, get_reconstruction_error


print(tf.__version__)

def load_encoder(encoder_path:str):
    assert(os.path.exists(encoder_path))
    assert(os.path.isdir(encoder_path))

    # Note: set 'compile' to False to avoid compile, which is required for training but not inference
    encoder = keras.models.load_model(os.path.join(encoder_path, "encoder"), compile=False)
    assert(encoder is not None)

    encoder.summary()
    encoder.compile()
    return encoder


def load_decoder(decoder_path: str):
    assert(os.path.exists(decoder_path))
    assert(os.path.isdir(decoder_path))

    decoder = keras.models.load_model(os.path.join(decoder_path, "decoder"), compile=False)
    assert(decoder is not None)

    decoder.summary()
    decoder.compile()
    return decoder


def load_model(encoder_path: str, decoder_path: str) -> tuple:

    encoder = load_encoder(encoder_path)
    decoder = load_decoder(decoder_path)
    return encoder, decoder


def normalize_images(imgs:np.ndarray) -> np.ndarray:
    """
    Normalizes a vector of images by its maxed value
    """
    return imgs / np.max(imgs)


def load_data(img_dim:int, channels:int, label_filepath: str) -> tuple:
    
    print('Loading data')
    dataset = RGB_Dataset()
    imgs, labels, orig_list = dataset.load_rgb(img_dim, channels, groundTruthFile=label_filepath)

    print('Normalizing data')
    normalised_input = normalize_images(imgs)

    return normalised_input, labels, orig_list


def generate_latent_space(encoder, imgs:np.ndarray) -> tuple:
    """
    Generates latent space from input list of images and encoder
    """

    assert(encoder is not None)
    assert(len(imgs) > 0)

    z_mean_list = []
    z_log_var_list = []
    z_list = []

    for img in tqdm.tqdm(imgs, desc='processing latent space'):
        img = np.expand_dims(img, axis=0)
        #TODO: Handle custom architectures -wmb
        _, _, _, _, _, z_mean, z_log_var, z = encoder.predict(img) # num of outputs must match num of outputs from network
        z_mean_list.append(np.array(z_mean))
        z_log_var_list.append(np.array(z_log_var))
        z_list.append(np.array(z))

    z_mean_results = np.squeeze(np.array(z_mean_list))
    z_log_var_results = np.squeeze(np.array((z_log_var_list)))
    z_results = np.squeeze(np.array(z_list))
    
    return z_mean_results, z_log_var_results, z_results


def write_latent_space(tb_path, now, z_mean_results, z_log_var_results, z_results, labels, orig_list) -> None:
    # save output to txt files in log
    print('Saving to text files.')
    
    prefix = os.path.abspath(os.path.join(tb_path, now))
    meta_path = os.path.abspath(os.path.join(tb_path, 'metadata.tsv'))

    np.savetxt((prefix +'_z_mean.txt'), z_mean_results, delimiter=',')
    np.savetxt((prefix +'_z_log_var.txt'), z_log_var_results, delimiter=',')
    np.savetxt((prefix +'z.txt'), z_results, delimiter=',')
    np.savetxt((meta_path), labels, delimiter='\t', fmt='%s')
    np.savetxt((prefix +'_z.tsv'), z_results, delimiter='\t', fmt='%s')

    # Make pandas data frame of results (assuming 32D Latent space)
    dataFrameName = prefix + 'dataFrame.csv'
    # TODO: Column name needs to be automated - wmb
    df = pd.DataFrame(z_results, columns=['z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7',
    'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15',
    'z16', 'z17', 'z18', 'z19', 'z20', 'z21', 'z22', 'z23',
    'z24', 'z25', 'z26', 'z27', 'z28', 'z29', 'z30', 'z31'])
    df2 = pd.DataFrame(orig_list)
    df = pd.concat([df2, df], axis=1)
    df.to_csv(dataFrameName, index=True)


def main(args):
    """
    Main Function
    """

    IMG_DIM = args.imgDim
    IMG_CH = args.imgCh
    GT_FILE = args.labels
    LOG_DIR = args.logDir
    PATH_TO_ENCODER = args.encoderPath
    PATH_TO_DECODER = args.decoderPath

    """
    ### Load Model
    """
    encoder, decoder = load_model(PATH_TO_ENCODER, PATH_TO_DECODER)
    
    """
    ## Import Training Data
    """
    imgs, labels, orig_list = load_data(IMG_DIM, IMG_CH, GT_FILE)

    now = "{:%Y%m%dT%H%M}".format(datetime.now())

    if (PATH_TO_DECODER == PATH_TO_ENCODER):
        tb_path = os.path.join(PATH_TO_DECODER, f"results_{now}")
    else:
        tb_path = os.path.join(LOG_DIR, f"results_{now}")

    # Create results directory
    # TODO: WindowsError is specific to Windows and does not exist on Linux
    # TODO: This exception has a logical flaw
    try:
        os.mkdir(tb_path)
    except WindowsError:
        os.mkdir(tb_path+'2')

    """
    ## Get latent space output from network
    """
    z_mean_results, z_log_var_results, z_results = generate_latent_space(encoder, imgs)

    write_latent_space(tb_path, now, z_mean_results, z_log_var_results, z_results, labels, orig_list)

    """
    ## Display how the latent space clusters different digit classes
    """

    # TODO: Move these functions to their own file, or use existing code
    def plot_label_clusters(encoder, decoder, data, labels, tb_path):
        print("Plot first 3 Latent Space Dim")
        # display a 2D plot of the digit classes in the latent space
        z_mean, z_log_var, z = encoder.predict(data)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=labels)
        #plt.colorbar()
        #plt.xlabel("z[0]")
        #plt.ylabel("z[1]")
        plt.savefig(tb_path+"/scatter.png", dpi=300)
        plt.show()

    # TODO: This can be thrown in a flag for plotting or its own script -wmb
    # plot_label_clusters(encoder, decoder, imgs, labels, tb_path)

    def generate_img_set(decoder, z, filepath, orig_list, orig_imgs, IMG_DIM, IMG_CH):
        print("Generating Latent Space Image Set")
        loss_per_img = []
        #TODO: Can use a join here - wmb
        path = filepath + "/generatedImgs/"
        #TODO: Path shouldn't exist, but check is still necessary - wmb
        os.mkdir(path)
        num_imgs = len(orig_list)
        for i in range(num_imgs):
            gt_img = orig_imgs[i]
            img_name = orig_list[i]['filename']

            # plt.figure()
            # ax = plt.axes([0, 0, 1, 1], frameon=False)
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            # plt.imshow(gt_img)
            # plt.autoscale(tight=True)
            # plt.show()

            #img = decoder(np.array([z[i]]))
            #TODO: Decoder is hardcoded for its outputs - wmb
            _, _, _, _, img = decoder(np.array([z[i]]))
            img = np.array(img)
            img = img[0, :, :, :]
            reconstruction_loss = get_reconstruction_error(gt_img[None, :, :, :], img[None, :, :, :])

            # ---Format to write out in 128x128 img format with openCV---#

            #TODO: Probably not all that important, but math on 255. float and round to integer - wmb
            img_rgb = 255.999 * img
            img_rgb = np.array(img_rgb, dtype='float32').astype('uint8')
            img_bgr = np.dstack((img_rgb[:, :, 2], img_rgb[:, :, 1], img_rgb[:, :, 0]))

            cv2.imwrite(path + img_name, img_bgr)

            # img = img[0].reshape(IMG_DIM, IMG_DIM, IMG_CH)

            # plt.figure()
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()
            # plt.close()

            loss_per_img.append(float(np.array(reconstruction_loss)))

        return loss_per_img

    loss_per_img = generate_img_set(decoder, z_results, tb_path, orig_list, imgs, IMG_DIM, IMG_CH)

    print("Saving Results to File")
    #TODO: Writing results to file are mixed all over the place, should consider placing them in this module - wmb
    #TODO: Write a single function that takes a dictionary and outputs all results to their respective files - wmb
    writeResults.simpleResults(tb_path, now+"results.csv", orig_list, loss_per_img)

    """
    ## Display a grid of sampled images
    """

    def plot_latent(encoder, decoder, IMG_CH, IMG_DIM, z, tb_path, loss_per_img):
        # display a n*n 2D manifold of digits

        import copy
        digit_size = IMG_DIM
        numImgSteps = 100

        #TODO: sorted is a python function, and actually can be used here instead of copy + sort - wmb
        sorted = copy.copy(loss_per_img)
        sorted.sort()


        for idx in range(0, z.shape[0]-1):

            z_vector1 = copy.copy(z[idx])
            z_vector2 = copy.copy(z[idx + 1])

            z_Step = (z_vector2 - z_vector1)/numImgSteps

            for n in range(0, numImgSteps+1):
                # z_sample = np.array([[xi, yi, zi, t, 0, 0, 0, 0]])
                fig_name = tb_path + "/latentSpace_x" + str(idx) + "_to_" + str(idx+1) + "_" + str(n) + ".png"
                z_vector = z_vector1 + n*z_Step

                x_decoded = decoder.predict(np.array([[z_vector]]))  # this is the image constructed by the network
                digit = x_decoded[4].reshape(digit_size, digit_size, IMG_CH) #  formatted for putting in the figure

                plt.figure()
                plt.axis('off')
                plt.tight_layout()
                plt.imshow(digit)
                plt.show()
                plt.close()

                img_rgb = 255.999 * digit
                img_rgb = np.array(img_rgb, dtype='float32').astype('uint8')
                img_bgr = np.dstack((img_rgb[:, :, 2], img_rgb[:, :, 1], img_rgb[:, :, 0]))
                cv2.imwrite(fig_name, img_bgr)

        z_vector1 = z[idx+1]
        z_vector2 = z[0]
        z_Step = (z_vector2 - z_vector1) / numImgSteps

        digit_size = IMG_DIM

        for n in range(0, numImgSteps + 1):
            # z_sample = np.array([[xi, yi, zi, t, 0, 0, 0, 0]])
            fig_name = tb_path + "/latentSpace_x" + str(idx+1) + "_to_" + str(0) + "_" + str(n) + ".png"
            z_vector = z_vector1 + n * z_Step

            #TODO: older code that expects single output from decoder - wmb
            x_decoded = decoder.predict(np.array([[z_vector]]))  # this is the image constructed by the network
            digit = x_decoded[4].reshape(digit_size, digit_size, IMG_CH)  # formatted for putting in the figure

            plt.figure()
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(digit)
            plt.show()
            plt.close()

            #TODO: Code is repeated here, throw in a function - wmb
            img_rgb = 255.999 * digit
            img_rgb = np.array(img_rgb, dtype='float32').astype('uint8')
            img_bgr = np.dstack((img_rgb[:, :, 2], img_rgb[:, :, 1], img_rgb[:, :, 0]))
            cv2.imwrite(fig_name, img_bgr)


    #plot_latent(encoder, decoder, IMG_CH, IMG_DIM, z_results, tb_path, loss_per_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, default = '.',
                        help="Ground Truth File with labels")
    parser.add_argument('--encoderPath', type=str, default='.',
                        help="path containing encoder")
    parser.add_argument('--decoderPath', type=str, default='.',
                        help="path containing decoder")
    parser.add_argument('--logDir', type=str, default='.',
                        help="path where logs will be saved")
    parser.add_argument('--imgDim', type=int, default=128)
    parser.add_argument('--imgCh', type=int, default=3)
    parser.add_argument('--latentDim', type=int, default=16)
    parser.add_argument('--imgDir', type=str, default="D:/AI_Learning_Images/snips/set2")
    parser.add_argument('--spriteImg', type=str, default=None)
    parser.add_argument('--finalImgSz', type=int, default=64)


    args = parser.parse_args()
    print(args)

    main(args)
