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

from Preprocess.tools.peakdetect import *
from Utils.utils import *

matplotlib.rcParams['font.size'] = 9


'''
This module is intended to segment a clear* handwriting image into words.
For this, the image is first split into lines.
Each line is then split into words.
Both methods make use of the peakdetect method.

*Prerequisites:
- image is greyscale binarised;
- image has no noise;
- image has no redundant information, except for the main component
    with the handwriten message;
- image is optimally rotated.
'''


# Functions


def segment_image_into_lines(image, line_peaks, output_directory):
    rows = []
    lines = []

# For the first peak
    for peak in line_peaks[0]:
        rows.append(peak[0])

    start = 0
    end = rows[0]
    line = image[start:end]
    lines.append(line)

    lines_directory = output_directory + '/lines0'
    ensure_directory(lines_directory)
    write_image(line, lines_directory + '/line_0.jpg')

    for idx in range (1, len(rows)):
        start = rows[idx-1]
        end = rows[idx]
        line = image[start:end]
        lines.append(line)

        write_image(line, lines_directory + '/line_' + str(idx) + '.jpg')

    idx = len(rows)-1
    start = rows[idx]
    end = len(image)
    line = image[start:end]
    lines.append(line)
    write_image(line, lines_directory + '/line_' + str(idx+1) + '.jpg')

    rows = []
    lines = []
# For the second peak
    for peak in line_peaks[1]:
        rows.append(peak[0])

    start = 0
    end = rows[0]
    line = image[start:end]
    lines.append(line)

    lines_directory = output_directory + '/lines1'
    ensure_directory(lines_directory)
    write_image(line, lines_directory + '/line_0.jpg')

    for idx in range (1, len(rows)):
        start = rows[idx-1]
        end = rows[idx]
        line = image[start:end]
        lines.append(line)

        write_image(line, lines_directory + '/line_' + str(idx) + '.jpg')

    idx = len(rows)-1
    start = rows[idx]
    end = len(image)
    line = image[start:end]
    lines.append(line)
    write_image(line, lines_directory + '/line_' + str(idx+1) + '.jpg')

    return lines


def segment_line_into_words(line_image):
    return ''


####### Main ########
#####################

def segment(image, line_peaks, output_directory):

    lines = segment_image_into_lines(image, line_peaks, output_directory)

#    line_counter = 0
#    for line in lines:
#        line_counter += 1
#        write_image(line, output_directory + '/line_' + str(line_counter))



    print ("Segmentation successful!")
