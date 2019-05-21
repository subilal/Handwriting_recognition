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

from .tools.peakdetect import *

matplotlib.rcParams['font.size'] = 9


def readImage(filename):
    inputImage = imread(filename)

    return inputImage


def constrast(image, method):
    if (method == 1): contrastedImage = adjust_gamma(image) 
    if (method == 2): contrastedImage = adjust_log(image)

    return contrastedImage


def binariseSauvola(image, window_size = 59, k = 0.5, r = 128):
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


# def getMainComponent(labeled_img_with_stats)
#     return ''

def getOptimumRotation(image, outputDirectory="Output"):
    rotDegree = 0
    for degree in range (-10, 10):
        rotatedImage = rotate (image, degree)
        writeImage(rotatedImage, outputDirectory+"/degree="+str(degree)+".jpg")

        # Create a model to quantify rotated images to determine the optimum one.


    return rotatedImage, rotDegree


# def trimImage(rotatedImage, blanks_allowed=10)
#     return ''


####### Main ########
#####################

def preprocess(inputImageName, outputDirectory):
    # Read image.
    inputImage = readImage(inputImageName)

    # Binarise image using both Sauvola and Otsu methods.    
    window_size = 25
    k = 0.5 # This is optional parameter
    r = 128 # This is optional parameter
    binarisedSauvola = binariseSauvola(inputImage, window_size=window_size, k=k, r=r)
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


    # First attempt, Otsu with connectivity 4:
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binarisedOtsu, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    labeled_img_with_stats = np.zeros(output.shape)
    labeled_img_with_stats[output == max_label] = 255

    writeImage(labeled_img_with_stats, outputDirectory+"/labeled_otsu_with_stats.jpg")


# Note: Connectivity 4 is better because, because this results in a better component,
# which then can be masked to isolate the main component.


    # # Second attempt, Otsu with connectivity 8:
    # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binarisedOtsu, connectivity=8)
    # sizes = stats[:, -1]
    # max_label = 1
    # max_size = sizes[1]
    # for i in range(2, nb_components):
    #     if sizes[i] > max_size:
    #         max_label = i
    #         max_size = sizes[i]
    # labeled_img_with_stats = np.zeros(output.shape)
    # labeled_img_with_stats[output == max_label] = 255

    # writeImage(labeled_img_with_stats, outputDirectory+"/labeled_otsu_with_stats_2.jpg")



    # Third attempt, Sauvola with connectivity 4:
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binarisedSauvola, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    labeled_img_with_stats2 = np.zeros(output.shape)
    labeled_img_with_stats2[output == max_label] = 255

    writeImage(labeled_img_with_stats2, outputDirectory+"/labeled_sauvola_with_stats.jpg")



    # # Fourth attempt, Sauvola with connectivity 8:
    # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binarisedSauvola, connectivity=8)
    # sizes = stats[:, -1]
    # max_label = 1
    # max_size = sizes[1]
    # for i in range(2, nb_components):
    #     if sizes[i] > max_size:
    #         max_label = i
    #         max_size = sizes[i]
    # labeled_img_with_stats2 = np.zeros(output.shape)
    # labeled_img_with_stats2[output == max_label] = 255

    # writeImage(labeled_img_with_stats2, outputDirectory+"/labeled_sauvola_with_stats_2.jpg")





    # Fifth attempt: negative Otsu based on labeled Otsu with stats.
    negativeImage = img_as_ubyte(labeled_img_with_stats / 255)
    negativeImage = 255 - negativeImage
    writeImage(negativeImage, outputDirectory+"/negativeImage2.jpg")

    print(negativeImage)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(negativeImage, connectivity=8)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    labeled_img_with_stats3 = np.zeros(output.shape)
    labeled_img_with_stats3[output == max_label] = 255

    writeImage(labeled_img_with_stats3, outputDirectory+"/labeled_otsu_negative.jpg")




    # Mask image 1: using labeled Otsu + negative labeled Otsu
    maskedOtsu = labeled_img_with_stats
    labeled_img_with_stats3 = img_as_ubyte(labeled_img_with_stats3 / 255)
    maskedOtsu[labeled_img_with_stats3 == 255] = 255

    writeImage(maskedOtsu, outputDirectory+"/maskedOtsu.jpg")



    # Mask image 2: using negative labeled Otsu and Sauvola binary
    maskedSauvola = binarisedSauvola
    maskedSauvola[labeled_img_with_stats3 == 255] = 255

    writeImage(maskedSauvola, outputDirectory+"/maskedSauvola.jpg")


    # Mask image 3: different Sauvola binary
    window_size = 59
    k = 0.5 # This is optional parameter
    r = 128 # This is optional parameter
    binarisedSauvola2 = binariseSauvola(inputImage, window_size=window_size, k=k, r=r)
    writeImage(binarisedSauvola2, outputDirectory+"/binarisedSauvola2.jpg")
    

    maskedSauvola = binarisedSauvola2
    maskedSauvola[labeled_img_with_stats3 == 255] = 255

    writeImage(maskedSauvola, outputDirectory+"/maskedSauvola2.jpg")


# So far, we isolated the image, the optimal parameters (currently) are:
#   - sauvola - window_size=25, k=0.5, r=128 -> used for masking.
#   - isolate componened based on Otsu binarised image.
#   - Connected components with 4 -> so that the connected component is just slightly larger
#            then the expcted, which will result in better isolation after masking wit Sauvola.
#   - Use maskedSauvola image.



# Find optimum rotation
    rotatedImage, rotDegree = getOptimumRotation(maskedSauvola)
    print ("Optimum rotation=" + str(rotDegree))

# # Optionally, trim image from blank lines to have the main component only.
# # (Use perhaps a blanks_allowed=10, i.e. have 10 white rows in each direction?)
#     preprocessedImage = trimImage(rotatedImage, blanks_allowed=10)
    


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
