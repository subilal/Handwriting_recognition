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

matplotlib.rcParams['font.size'] = 9


'''
This module is intended to clear the image containing the handwriten message
and to isolate the main component, i.e. the actual handwriting.

Steps:
- image is first binarised;
- then the main component is identified and isolated;
- background is cleared of noise;
- otimum rotation is determined.
'''

def constrast(image, method):
    if (method == 1): contrastedImage = adjust_gamma(image) 
    if (method == 2): contrastedImage = adjust_log(image)

    return contrasted_image


def binarise_sauvola(image, window_size = 59, k = 0.5, r = 128):
    thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=k, r=r)

    binarised_sauvola = image > thresh_sauvola
    binarised_sauvola = img_as_ubyte(binarised_sauvola)

    return binarised_sauvola


def binarise_otsu(image):
    binarised_otsu = image > threshold_otsu(image)
    binarised_otsu = img_as_ubyte(binarised_otsu)
    
    return binarised_otsu


def get_optimum_rotation(image, output_directory, lookahead=30, min_degree=-10, max_degree=10):
    optimum_rot_degree = -90
    optimum_score = 0
    optimum_rot_image = image
    optimum_rot_line_peaks = []

    for degree in range (min_degree, max_degree):

        rotated_image = rotate(image, degree, resize=False, cval=1, mode ='constant')
        
        # Rotate results in a normalised floating image -> convert it to uint8
        rotated_image = rotated_image * 255
        rotated_image = rotated_image.astype(np.uint8)
        write_image(rotated_image, output_directory+'/rotated_' + str(degree) + '.jpg')

        # 1 = column reduction.
        # CV_REDUCE_AVG instead of sum, because we want the normalized number of pixels
        histogram = cv2.reduce(rotated_image, 1, cv2.REDUCE_AVG)
        # Transpose column vector into row vector
        histogram = histogram.reshape(-1)

        plt.plot(histogram)
        plt.title('Degree=' + str(degree))
        plt.savefig(output_directory+'/histogram_' + str(degree) + '.jpg')
        plt.clf()

        line_peaks = peakdetect(histogram, lookahead=lookahead)

        # Note: It is expected that the algorithm will give an equal number
        # of positive and negative peaks.
        number_peaks = len(line_peaks[0])
        for peak in range(0, number_peaks):
            score = line_peaks[0][peak][1] - line_peaks[1][peak][1]
        score = score / number_peaks

        print ('Degree=' + str(degree) + '; Score=' + str(score))

        if score >= optimum_score and abs(degree) <= abs(optimum_rot_degree):
            optimum_score = score
            optimum_rot_degree = degree
            optimum_rot_image = rotated_image
            optimum_rot_line_peaks = line_peaks

    return optimum_rot_image, optimum_rot_line_peaks, optimum_rot_degree


# Rotate image with border. Credit:
# https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


####### Main ########
#####################

def preprocess(input_image_name, output_directory):
    # Read image.
    input_image = read_image(input_image_name)

    # Binarise image using both Sauvola and Otsu methods.    
    window_size = 25
    k = 0.5 # This is optional parameter
    r = 128 # This is optional parameter
    binarised_sauvola = binarise_sauvola(input_image, window_size=window_size, k=k, r=r)
    binarised_otsu = binarise_otsu(input_image)

    # Save binarised images.
    write_image(binarised_sauvola, output_directory+"/binarisedSauvola.jpg")
    write_image(binarised_otsu, output_directory+"/binarisedOtsu.jpg")


    # Get the connected componenets. Get a feeling of how the connection is done.
    # Credit: https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
    ret, labels = cv2.connectedComponents(binarised_otsu)
    
    # Map component labels to hue val.
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    
    # cvt to BGR for display.
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    
    # set bg label to black.
    labeled_img[label_hue == 0] = 0
    
    write_image(labeled_img, output_directory+"/labeled_otsu.jpg")


    # First attempt, Otsu with connectivity 4:
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binarised_otsu, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    labeled_img_with_stats = np.zeros(output.shape)
    labeled_img_with_stats[output == max_label] = 255

    write_image(labeled_img_with_stats, output_directory+"/labeled_otsu_with_stats.jpg")


# Note: Connectivity 4 is better because, because this results in a better component,
# which then can be masked to isolate the main component.


    # # Second attempt, Otsu with connectivity 8:
    # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binarised_otsu, connectivity=8)
    # sizes = stats[:, -1]
    # max_label = 1
    # max_size = sizes[1]
    # for i in range(2, nb_components):
    #     if sizes[i] > max_size:
    #         max_label = i
    #         max_size = sizes[i]
    # labeled_img_with_stats = np.zeros(output.shape)
    # labeled_img_with_stats[output == max_label] = 255

    # write_image(labeled_img_with_stats, output_directory+"/labeled_otsu_with_stats_2.jpg")



    # Third attempt, Sauvola with connectivity 4:
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binarised_sauvola, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    labeled_img_with_stats2 = np.zeros(output.shape)
    labeled_img_with_stats2[output == max_label] = 255

    write_image(labeled_img_with_stats2, output_directory+"/labeled_sauvola_with_stats.jpg")



    # # Fourth attempt, Sauvola with connectivity 8:
    # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binarised_sauvola, connectivity=8)
    # sizes = stats[:, -1]
    # max_label = 1
    # max_size = sizes[1]
    # for i in range(2, nb_components):
    #     if sizes[i] > max_size:
    #         max_label = i
    #         max_size = sizes[i]
    # labeled_img_with_stats2 = np.zeros(output.shape)
    # labeled_img_with_stats2[output == max_label] = 255

    # write_image(labeled_img_with_stats2, output_directory+"/labeled_sauvola_with_stats_2.jpg")





    # Fifth attempt: negative Otsu based on labeled Otsu with stats.
    negative_image = img_as_ubyte(labeled_img_with_stats / 255)
    negative_image = 255 - negative_image
    write_image(negative_image, output_directory+"/negativeImage2.jpg")

   
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(negative_image, connectivity=8)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    labeled_img_with_stats3 = np.zeros(output.shape)
    labeled_img_with_stats3[output == max_label] = 255

    write_image(labeled_img_with_stats3, output_directory+"/labeled_otsu_negative.jpg")




    # Mask image 1: using labeled Otsu + negative labeled Otsu
    masked_otsu = labeled_img_with_stats
    labeled_img_with_stats3 = img_as_ubyte(labeled_img_with_stats3 / 255)
    masked_otsu[labeled_img_with_stats3 == 255] = 255

    write_image(masked_otsu, output_directory+"/maskedOtsu.jpg")



    # Mask image 2: using negative labeled Otsu and Sauvola binary
    masked_sauvola = binarised_sauvola
    masked_sauvola[labeled_img_with_stats3 == 255] = 255

    write_image(masked_sauvola, output_directory+"/maskedSauvola.jpg")


    # Mask image 3: different Sauvola binary
    window_size = 59
    k = 0.5 # This is optional parameter
    r = 128 # This is optional parameter
    binarised_sauvola_2 = binarise_sauvola(input_image, window_size=window_size, k=k, r=r)
    write_image(binarised_sauvola_2, output_directory+"/binarisedSauvola2.jpg")
    

    masked_sauvola = binarised_sauvola_2
    masked_sauvola[labeled_img_with_stats3 == 255] = 255

    write_image(masked_sauvola, output_directory+"/maskedSauvola2.jpg")


# So far, we isolated the image, the optimal parameters (currently) are:
#   - sauvola - window_size=25, k=0.5, r=128 -> used for masking.
#   - isolate componened based on Otsu binarised image.
#   - Connected components with 4 -> so that the connected component is just slightly larger
#            then the expcted, which will result in better isolation after masking wit Sauvola.
#   - Use maskedSauvola image.



# Find optimum rotation
    rotation_directory = output_directory + '/rotated'
    remove_directory(rotation_directory)
    ensure_directory(rotation_directory)
    lookahead = 20
    rot_image, rot_line_peaks, rot_degree = get_optimum_rotation(masked_sauvola, rotation_directory, lookahead = lookahead)


# # Optionally, trim image from blank lines to have the main component only.
# # (Use perhaps a blanks_allowed=10, i.e. have 10 white rows in each direction?)
#     preprocessedImage = trimImage(rotatedImage, blanks_allowed=10)
    
    print ("Preprocessing successful!")

    return rot_image, rot_line_peaks, rot_degree
