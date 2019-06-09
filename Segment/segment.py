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

from Utils.peakdetect import *
from Utils.io import *
from Utils.filesystem import *
from Utils.trim_image import *

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
    height = image.shape[0]

    if (not line_peaks[0] and not line_peaks[1]):
        print ('Line_peaks is empty')
        return words
    elif (line_peaks[0]):
        for peak in line_peaks[0]:
            rows.append(peak[0])

    rows.append(height)

    start = 0
    end = rows[0]
    line = image[start:end]
    lines.append(line)

    remove_directory(output_directory)
    ensure_directory(output_directory)
    write_image(line, output_directory + '/line_0.jpg')

    for idx in range (1, len(rows)):
        start = rows[idx-1]
        end = rows[idx]
        line = image[start:end]
        lines.append(line)

        write_image(line, output_directory + '/line_' + str(idx) + '.jpg')

    return lines


def segment_line_into_words(line_image, line_idx, word_peaks, output_directory):
    cols = []
    words = []

    height = line_image.shape[0]
    width = line_image.shape[1]

    if (not word_peaks[0] and not word_peaks[1]):
        print ('Word_peaks is empty')
        return words
    elif (word_peaks[0]):
        for peak in word_peaks[0]:
            cols.append(peak[0])
        
    cols.append(width)

    print (cols)

    start = 0
    end = cols[0]
    height = line_image.shape[0]
    word = line_image[0:height, start:end]
    words.append(word)

    remove_directory(output_directory)
    ensure_directory(output_directory)
    write_image(word, output_directory + '/word_0.jpg')

    for idx in range (1, len(cols)):
        start = cols[idx-1]
        end = cols[idx]
        word = line_image[0:height, start:end]
        words.append(word)

        write_image(word, output_directory + '/word_' + str(idx) + '.jpg')

    return words



####### Main ########
#####################

def segment(image, line_peaks, output_directory):
    lines_directory = output_directory + '/lines'
    lines = segment_image_into_lines(image, line_peaks, lines_directory)

    print ("Lines created!")

    line_idx = 0
    for line in lines:
        line_histogram = cv2.reduce(line, 0, cv2.REDUCE_AVG)
#        trim_image (line, 0)
        # Transpose column vector into row vector
        line_histogram = line_histogram.reshape(-1)
        lookahead = 30
        word_peaks = peakdetect(line_histogram, lookahead = lookahead)

        print (line_histogram)
        print (word_peaks)
        plt.plot(line_histogram)
        plt.title('Line=' + str(line_idx))
        plt.savefig(output_directory+'/histogram_' + str(line_idx) + '.jpg')
        plt.clf()


        words_directory = lines_directory + '/line_' + str(line_idx)
        remove_directory(words_directory)
        ensure_directory(words_directory)

        words = segment_line_into_words(line, line_idx, word_peaks, words_directory)
        print ("Words for line " + str(line_idx) + " created!")

        line_idx = line_idx + 1


    print ("Segmentation successful!")
