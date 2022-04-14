#!/usr/bin/env python

"""
## Setup
"""

import argparse
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
from VAE.utils.VAE_utils import RGB_Dataset


print(tf.__version__)

def main():
    """
    Main Function
    """

    IMG_DIM = args.imgDim
    IMG_CH = args.imgCh
    GT_FILE = args.labels
    LOG_DIR = args.logDir
    PATH_TO_ENCODER = args.encoderPath
    PATH_TO_DECODER = args.decoderPath
    UMAP_FILE = args.umap

    """
    ### Load Model
    """
    # Note: set 'compile' to False to avoid compile, which is required for training but not inference
    encoder = keras.models.load_model(os.path.join(PATH_TO_ENCODER, "encoder"), compile=False)
    encoder.summary()
    encoder.compile()

    decoder = keras.models.load_model(os.path.join(PATH_TO_DECODER, "decoder"), compile=False)
    decoder.summary()
    decoder.compile()

    """
    ## Import Training Data
    """
    print('Loading data')
    dataset = RGB_Dataset()
    imgs, labels, orig_list = dataset.load_rgb(IMG_DIM, IMG_CH, groundTruthFile=GT_FILE)

    print('Normalizing data')
    normalised_input = (imgs) / np.max(imgs)
    imgs = normalised_input

    now = "{:%Y%m%dT%H%M}".format(datetime.now())

    if (PATH_TO_DECODER == PATH_TO_ENCODER):
        tb_path = os.path.join(PATH_TO_DECODER, f"umap_results_{now}")
    else:
        tb_path = os.path.join(LOG_DIR, f"umap_results_{now}")

    # Create results directory
    # TODO: WindowsError is specific to Windows and does not exist on Linux
    # TODO: This exception has a logical flaw
    try:
        os.mkdir(tb_path)
    except WindowsError:
        os.mkdir(tb_path+'2')

    """
    ## Get latent space embedding from UMAP file
    """

    umap_data = pd.read_csv(UMAP_FILE, header=None, index_col=0, squeeze=True).to_numpy()
    umap_embeddings = umap_data[1:522, 3:35]


    z_mean_list = []
    z_log_var_list = []
    z_list = []

    # for img in tqdm.tqdm(imgs, desc='processing latent space'):
    #     img = np.expand_dims(img, axis=0)
    #     _, _, _, _, _, z_mean, z_log_var, z = encoder.predict(img) # num of outputs must match num of outputs from network
    #     z_mean_list.append(np.array(z_mean))
    #     z_log_var_list.append(np.array(z_log_var))
    #     z_list.append(np.array(z))

    # z_mean_results = np.squeeze(np.array(z_mean_list))
    # z_log_var_results = np.squeeze(np.array((z_log_var_list)))
    # z_results = np.squeeze(np.array(z_list))

    # save output to txt files in log
    print('Saving to text files.')
    # TODO: Joins need to be properly resolved
    # np.savetxt((tb_path + '/'+ now +'_z_mean.txt'), z_mean_results, delimiter=',')
    # np.savetxt((tb_path + '/'+ now +'_z_log_var.txt'), z_log_var_results, delimiter=',')
    # np.savetxt((tb_path + '/'+ now +'z.txt'), z_results, delimiter=',')
    # np.savetxt((tb_path + '/' +'metadata.tsv'), labels, delimiter='\t', fmt='%s')
    # np.savetxt((tb_path+'/'+now+'_z.tsv'), z_results, delimiter='\t', fmt='%s')

    # Make pandas data frame of results (assuming 32D Latent space)
    # dataFrameName = tb_path + '/' + now + 'dataFrame.csv'
    # df = pd.DataFrame(z_results, columns=['z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7',
    # 'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15',
    # 'z16', 'z17', 'z18', 'z19', 'z20', 'z21', 'z22', 'z23',
    # 'z24', 'z25', 'z26', 'z27', 'z28', 'z29', 'z30', 'z31'])
    # df2 = pd.DataFrame(orig_list)
    # df = pd.concat([df2, df], axis=1)
    # df.to_csv(dataFrameName, index=True)

    """
    ## Display how the latent space clusters different digit classes
    """

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

    # plot_label_clusters(encoder, decoder, imgs, labels, tb_path)

    def getImageLoss(gt_img, img):
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(gt_img[None, :, :, :], img[None, :, :, :])
        )
        #reconstruction_loss *= IMG_DIM * IMG_DIM

        return reconstruction_loss

    def generate_img_set(decoder, z, filepath, orig_list, orig_imgs, IMG_DIM, IMG_CH):
        print("Generating Latent Space Image Set")
        loss_per_img = []
        path = filepath + "/generatedImgs/"
        os.mkdir(path)
        num_imgs = len(orig_list)
        for i in tqdm.tqdm(range(num_imgs), desc="Generating Image Set"):
            gt_img = orig_imgs[i]
            img_name = orig_list[i]['file_name']

            # plt.figure()
            # ax = plt.axes([0, 0, 1, 1], frameon=False)
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            # plt.imshow(gt_img)
            # plt.autoscale(tight=True)
            # plt.show()

            #img = decoder(np.array([z[i]]))
            _, _, _, _, img = decoder(np.asarray([z[i]]).astype('float32'))
            img = np.array(img)
            img = img[0, :, :, :]
            reconstruction_loss = getImageLoss(gt_img, img)

            # ---Format to write out in 128x128 img format with openCV---#

            img_rgb = 255.999 * img
            img_rgb = np.array(img_rgb, dtype='float32').astype('uint8')
            img_bgr = np.dstack((img_rgb[:, :, 2], img_rgb[:, :, 1], img_rgb[:, :, 0]))

            cv2.imwrite(os.path.join(filepath, "generatedImgs", img_name), img_bgr)

            # img = img[0].reshape(IMG_DIM, IMG_DIM, IMG_CH)

            # plt.figure()
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()
            # plt.close()

            loss_per_img.append(float(np.array(reconstruction_loss)))

        return loss_per_img

    loss_per_img = generate_img_set(decoder, umap_embeddings, tb_path, orig_list, imgs, IMG_DIM, IMG_CH)

    print("Saving Results to File")
    writeResults.simpleResults(tb_path, now+"umap_results.csv", orig_list, loss_per_img)

    """
    ## Display a grid of sampled images
    """

    def plot_latent(encoder, decoder, IMG_CH, IMG_DIM, z, tb_path, loss_per_img):
        # display a n*n 2D manifold of digits

        import copy
        digit_size = IMG_DIM
        numImgSteps = 10

        sorted = copy.copy(loss_per_img)
        sorted.sort()


        for idx in np.arange(0, z.shape[0]-1, 10):

            z_vector1 = copy.copy(z[idx])
            if idx+numImgSteps < z.shape[0] - 1:
                z_vector2 = copy.copy(z[idx + numImgSteps])
            else:
                z_vector2 = copy.copy(z[z.shape[0]-1])

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

            x_decoded = decoder.predict(np.array([[z_vector]]))  # this is the image constructed by the network
            digit = x_decoded[4].reshape(digit_size, digit_size, IMG_CH)  # formatted for putting in the figure

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
    parser.add_argument('--umap', type=str, default='.')


    args = parser.parse_args()
    print(args)

    main()
