import sys, os

import cv2

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import warnings

from skimage import img_as_float, img_as_ubyte
from skimage.exposure import adjust_gamma, adjust_log
from skimage.io import imread, imsave, imshow
from skimage.filters import threshold_sauvola, threshold_otsu
from skimage.transform import rotate

from Utils.io import *
from Utils.filesystem import *
from Utils.trim_image import *
from Utils.peakdetect import *

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
    if method == 1: contrastedImage = adjust_gamma(image) 
    if method == 2: contrastedImage = adjust_log(image)

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


def get_optimum_rotation(image, output_directory, lookahead=30, min_degree=-10, max_degree=10, runmode=1):
    optimum_rot_degree = -90
    optimum_score = 0
    optimum_rot_image = image
    optimum_rot_line_peaks = []

    for degree in range (min_degree, max_degree):

        rotated_image = rotate(image, degree, resize=False, cval=1, mode ='constant')
        
        # Rotate results in a normalised floating image -> convert it to uint8
        rotated_image = rotated_image * 255
        rotated_image = rotated_image.astype(np.uint8)
        
        if runmode > 0: # Show intermediary rotated images in normal and debug mode
            write_image(rotated_image, output_directory + '/rotated_' + str(degree) + '.jpg', runmode=runmode)

        # 1 = column reduction.
        # CV_REDUCE_AVG instead of sum, because we want the normalized number of pixels
        histogram = cv2.reduce(rotated_image, 1, cv2.REDUCE_AVG)
        # Transpose column vector into row vector
        histogram = histogram.reshape(-1)

        if (runmode > 0): # Show intermediary histograms in normal and debug mode
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

        if runmode > 1: # Show tested degrees in debug mode only
            print ('Degree=' + str(degree) + '; Score=' + str(score))

        if score >= optimum_score and abs(degree) <= abs(optimum_rot_degree):
            optimum_score = score
            optimum_rot_degree = degree
            optimum_rot_image = rotated_image
            optimum_rot_line_peaks = line_peaks

    return optimum_rot_image, optimum_rot_line_peaks, optimum_rot_degree


####### Main ########
#####################

def preprocess(input_image_name, output_directory, runmode=1):
    # Read image.
    input_image = read_image(input_image_name)

    # Binarise image using both Sauvola and Otsu methods.    
    window_size = 25
    k = 0.5 # This is optional parameter
    r = 128 # This is optional parameter
    binarised_sauvola = binarise_sauvola(input_image, window_size=window_size, k=k, r=r)
    binarised_otsu = binarise_otsu(input_image)

    # Save binarised images.
    write_image(binarised_sauvola, output_directory + "/binarisedSauvola.jpg", runmode=runmode)
    write_image(binarised_otsu, output_directory + "/binarisedOtsu.jpg", runmode=runmode)


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
    
    write_image(labeled_img, output_directory + "/labeled_otsu.jpg", runmode=runmode)


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

    write_image(labeled_img_with_stats, output_directory + "/labeled_otsu_with_stats.jpg", runmode=runmode)


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

    # write_image(labeled_img_with_stats, output_directory + "/labeled_otsu_with_stats_2.jpg", runmode=runmode)



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

    write_image(labeled_img_with_stats2, output_directory + "/labeled_sauvola_with_stats.jpg", runmode=runmode)



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

    # write_image(labeled_img_with_stats2, output_directory + "/labeled_sauvola_with_stats_2.jpg", runmode=runmode)





    # Fifth attempt: negative Otsu based on labeled Otsu with stats.
    if runmode < 2: # Surpress the errors if program is not in debug mode
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            negative_image = img_as_ubyte(labeled_img_with_stats / 255)
    else:
        negative_image = img_as_ubyte(labeled_img_with_stats / 255)

    negative_image = 255 - negative_image
    write_image(negative_image, output_directory + "/negativeImage2.jpg", runmode=runmode)

   
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

    write_image(labeled_img_with_stats3, output_directory + "/labeled_otsu_negative.jpg", runmode=runmode)




    # Mask image 1: using labeled Otsu + negative labeled Otsu
    masked_otsu = labeled_img_with_stats

    if runmode < 2: # Surpress the errors if program is not in debug mode
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labeled_img_with_stats3 = img_as_ubyte(labeled_img_with_stats3 / 255)
    else:
       labeled_img_with_stats3 = img_as_ubyte(labeled_img_with_stats3 / 255)

    masked_otsu[labeled_img_with_stats3 == 255] = 255
    write_image(masked_otsu, output_directory + "/maskedOtsu.jpg", runmode=runmode)



    # Mask image 2: using negative labeled Otsu and Sauvola binary
    masked_sauvola = binarised_sauvola
    masked_sauvola[labeled_img_with_stats3 == 255] = 255

    write_image(masked_sauvola, output_directory + "/maskedSauvola.jpg", runmode=runmode)


    # Mask image 3: different Sauvola binary
    window_size = 59
    k = 0.5 # This is optional parameter
    r = 128 # This is optional parameter
    binarised_sauvola_2 = binarise_sauvola(input_image, window_size=window_size, k=k, r=r)
    write_image(binarised_sauvola_2, output_directory + "/binarisedSauvola2.jpg", runmode=runmode)
    

    masked_sauvola = binarised_sauvola_2
    masked_sauvola[labeled_img_with_stats3 == 255] = 255

    write_image(masked_sauvola, output_directory + "/maskedSauvola2.jpg", runmode=runmode)


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
    min_degree = -6
    max_degree = 6
    rot_image, rot_line_peaks, rot_degree = get_optimum_rotation(masked_sauvola, rotation_directory, lookahead = lookahead,
                                                                min_degree=min_degree, max_degree=max_degree, runmode=runmode)

    if runmode > 0:
        filename = "optimumRotation=" + str(rot_degree) + ".jpg"
        filepath = os.path.join(output_directory, filename)
        write_image(rot_image, filepath, runmode=runmode)
    
    if runmode > 1:
        print ("Optimum rotation = " + str(rot_degree))
        write_line_peaks(rot_line_peaks, output_directory + '/line_peaks')

# # Optionally, trim image from blank lines to have the main component only.
# # (Use perhaps a blanks_allowed=10, i.e. have 10 white rows in each direction?)
#     preprocessedImage = trimImage(rotatedImage, blanks_allowed=10)

    padding = 50
    trimmed_rot_image = trim_image_with_component(rot_image, padding=padding)

    if runmode > 0:
        filename = "trimmed_" + filename
        filepath = os.path.join(output_directory, filename)
        write_image(trimmed_rot_image, filepath, runmode=runmode)

    return trimmed_rot_image
