import sys, os

import cv2

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from skimage import img_as_float, img_as_ubyte
from skimage.exposure import adjust_gamma, adjust_log
from skimage.io import imread, imsave, imshow
from skimage.filters import threshold_sauvola, threshold_otsu

os.path.join(os.path.abspath(__file__))
from .tools.peakdetect import *

matplotlib.rcParams['font.size'] = 9


def readImage(filename):
    inputImage = imread(filename)

    return inputImage


def constrast(image, method):
    if (method == 1): contrastedImage = adjust_gamma(image) 
    if (method == 2): contrastedImage = adjust_log(image)

    return contrastedImage


def binariseSauvola(image):
    window_size = 25
    k = 0.5 # This is optional parameter
    r = 128 # This is optional parameter
    thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=k, r=r)

    binarised_sauvola = image > thresh_sauvola
    binarised_sauvola = img_as_ubyte(binarised_sauvola)

    return binarised_sauvola


def binariseOtsu(image):
    binarised_otsu = image > threshold_otsu(image)
    binarised_otsu = img_as_ubyte(binarised_otsu)
    
    return binarised_otsu


def writeImage(image, filename):
    path_output = os.path.abspath(filename)
    imsave(fname=path_output,arr=image)




####### Main ########
#####################

def preprocess(inputImageName, outputDirectory):
    # Read image.
    inputImage = readImage(inputImageName)

    # Binarise image using both Sauvola and Otsu methods.
    binarisedSauvola = binariseSauvola(inputImage)
    binarisedOtsu = binariseOtsu(inputImage)

    # Save binarised images.
    writeImage(binarisedSauvola, outputDirectory+"/binarisedSauvola.jpg")
    writeImage(binarisedOtsu, outputDirectory+"/binarisedOtsu.jpg")


    # Get the connected componenets
    # Credit: https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
    ret, labels = cv2.connectedComponents(binarisedOtsu)

    # Map component labels to hue val.
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display.
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black.
    labeled_img[label_hue == 0] = 0

    writeImage(labeled_img, outputDirectory+"/labeled_otsu.jpg")


    # Second attempt:
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binarisedOtsu, connectivity=4)
#    print (nb_components)
#    print (output)
#    print (stats)
#    print (centroids)

    # Get the largest component.
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    labeled_img_with_stats = np.zeros(output.shape)
    labeled_img_with_stats[output == max_label] = 255

#    labeled_img_with_stats = img_as_ubyte(labeled_img_with_stats)
    writeImage(labeled_img_with_stats, outputDirectory+"/labeled_otsu_with_stats.jpg")



    print ("Preprocessing successful!")

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
    preprocess(inputImageName, outputDirectory)
