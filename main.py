import sys, os

from Preprocess.preprocess import *


input_image = "Input/P168-Fg016-R-C01-R01-fused.jpg"
output_directory = "Output"

# First, preprocess images.
preprocess(input_image,output_directory)
