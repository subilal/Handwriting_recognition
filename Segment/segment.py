import sys, os

import cv2

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from skimage import img_as_float, img_as_ubyte
from skimage.exposure import adjust_gamma, adjust_log
from skimage.io import imread, imsave, imshow
from skimage.filters import threshold_sauvola, threshold_otsu
from skimage.transform import rotate

matplotlib.rcParams['font.size'] = 9


# Functions

def segmentImageIntoLines(image):
    return ''


def segmentLineIntoWords(lineImage):
    return ''


####### Main ########
#####################

def segment(inputImageName, outputDirectory):
    # Read image.
    inputImage = readImage(inputImageName)

   

    print ("Segmentation successful!")

    ##########################



    # Show all preprocessing steps

    # plt.figure(figsize=(8, 3))

    # plt.subplot(1, 2, 1)
    # plt.imshow(inputImage, cmap=plt.cm.gray)
    # plt.title('Original')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(binarisedImage, cmap=plt.cm.gray)
    # plt.title('Sauvola Threshold')
    # plt.axis('off')


    # plt.show()

if __name__ == "__main__":
    inputImageName = sys.argv[1]
    outputDirectory = sys.argv[2]
    segment(inputImageName, outputDirectory)
