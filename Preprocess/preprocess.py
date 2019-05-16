import sys
import os

import matplotlib
import matplotlib.pyplot as plt

from skimage import img_as_float, img_as_ubyte
from skimage.exposure import adjust_gamma, adjust_log
from skimage.io import imread, imsave, imshow
from skimage.filters import threshold_sauvola

from tools.peakdetect import *

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

    return binary_sauvola


def writeBinarisedImage(image, filename):

    path_output = os.path.abspath(filename)
    
    imsave(fname=path_output,arr=image)





def segment(image):
    max_peaks, min_peaks = peakdetect(image, lookahead=40)
    print(min_peaks)
    return ''


#    peak = []
#    for y in peaks[0]:
##        peak.append(y[0])
  #      # plt.plot(y[0], y[1], "r*")
   #     cv2.line(rotated2, (0, y[0]), (W, y[0]), (255, 0, 0), 3)



####### Main ########
#####################

inputImageName = sys.argv[1]
outputImageName = sys.argv[2]


# Read name
inputImage = readImage(inputImageName)

# Binarise using Sauvola binarisation with threshold=0.5
binarisedImage = binarise (inputImage)
binarisedImage = img_as_ubyte(binarisedImage)

# Save binarised image
writeBinarisedImage(binarisedImage, outputImageName)

##########################

### Segment it ###

segmentedImage = segment(binarisedImage)





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
