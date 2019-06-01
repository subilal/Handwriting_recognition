from Segment.segment import *

'''
This file is intended only for runnning and testing
The segmentation step.
'''

input_image_name = "TestInputSegment/image.jpg"
input_line_peaks = "TestInputSegment/line_peaks.npz"
output_directory = "TestOutputSegment"

image = read_image (input_image_name)
line_peaks = read_line_peaks (input_line_peaks)

segment(image, line_peaks, output_directory)

print ('Main Segmentation successful')
