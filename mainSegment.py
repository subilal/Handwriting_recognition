from Segment.segment import *

'''
This file is intended only for runnning and testing
The segmentation step.
'''

image_name = "image.jpg"
line_peaks_file = "line_peaks.npz"

input_directory = "TestInputSegment"
output_directory = "TestOutputSegment"

input_image_name = os.path.join(input_directory, image_name)
input_line_peaks = os.path.join(input_directory, line_peaks_file)

image = read_image (input_image_name)
line_peaks = read_line_peaks (input_line_peaks)

print ("")
print ("Segmenting image " + image_name)
print ("")

segment(image, line_peaks, output_directory, runmode=2)

print ("")
print ("Finished Segmenting image " + image_name)
print ("--------------------------------------------")
