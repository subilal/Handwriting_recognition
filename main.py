import sys, os

from Preprocess.preprocess import *
from Segment.segment import *


# First, preprocess images.
input_image_name = "Input/P168-Fg016-R-C01-R01-fused.jpg"
output_directory = "Output"

rot_image, rot_line_peaks, rot_degree = preprocess(input_image_name, output_directory)
segment(rot_image, rot_line_peaks, output_directory)
