import sys, os

import numpy as np
import warnings

from skimage.io import imread, imsave

'''
This file is intended to handle the I/O flows.
'''

# Reading

def read_image(filename):
    image = imread(filename)

    return image

def read_line_peaks(filename):
    line_peaks_data = np.load(filename)

    line_peaks0 = line_peaks_data['line_peaks0'].tolist()
    line_peaks1 = line_peaks_data['line_peaks1'].tolist()

    line_peaks = [line_peaks0, line_peaks1]

    return line_peaks


# Writing

def write_image(image, filename, runmode=1):
    path_output = os.path.abspath(filename)
    image = image.astype(np.uint8)
    if runmode < 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(fname=path_output,arr=image)
    else:
        imsave(fname=path_output,arr=image)


def write_line_peaks(line_peaks, filename):
    path_output = os.path.abspath(filename)
    np.savez(filename, line_peaks0=line_peaks[0], line_peaks1=line_peaks[1])
