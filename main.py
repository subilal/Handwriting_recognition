import sys, os

from Preprocess.preprocess import *


inputImage = "Input/P168-Fg016-R-C01-R01-fused.jpg"
outputDirectory = "Output"

# First, preprocess images.
preprocess(inputImage,outputDirectory)
