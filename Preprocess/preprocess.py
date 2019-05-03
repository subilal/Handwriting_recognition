import sys
import os

import matplotlib
import matplotlib.pyplot as plt

from skimage import img_as_float
from skimage.exposure import adjust_gamma, adjust_log
from skimage.io import imread, imsave, imshow
from skimage.filters import threshold_sauvola


matplotlib.rcParams['font.size'] = 9



def readImage(filename):
    inputImage = imread(filename)

    return inputImage


def constrast(image, method):

    if (method == 1): contrastedImage = adjust_gamma(image) 
    if (method == 2): contrastedImage = adjust_log(image)

    return contrastedImage


def binarise (image):
    window_size = 25
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)

    binary_sauvola = image > thresh_sauvola

    return img_as_float(binary_sauvola)


def writeBinarisedImage(image, filename):

    path_output = os.path.abspath(filename)
    
    imsave(fname=path_output,arr=image)


# Main

inputImageName = sys.argv[1]
outputImageName = sys.argv[2]


# Read name
inputImage = readImage(inputImageName)

# Binarise using Sauvola binarisation with threshold=0.5
binarisedImage = binarise (inputImage)


# Show all preprocessing steps

plt.figure(figsize=(8, 3))

plt.subplot(1, 2, 1)
plt.imshow(inputImage, cmap=plt.cm.gray)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binarisedImage, cmap=plt.cm.gray)
plt.title('Sauvola Threshold')
plt.axis('off')


plt.show()
