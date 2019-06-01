import sys, os

from Preprocess.preprocess import *
from Segment.segment import *


# First, preprocess images.
input_image = "Input/P168-Fg016-R-C01-R01-fused.jpg"
output_directory = "Output"
preprocess(input_image,output_directory)

# Second, segment images.
input_image = "Output/optimumRotation.jpg"
output_directory = "Output/segmented"
segment(input_image,output_directory)
