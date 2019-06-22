import sys, os

import argparse
import glob

import time

from Preprocess.preprocess import *
from Segment.segment import *
from Classifier.classifier import *


#################
### Arguments ###
#################

# Parse arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-d", "--debug", help="run program in debug mode",
									action="store_true")
group.add_argument("-f", "--fast",  help="skip over intermediary results (used to speed-up the program)",
									action="store_true")
parser.add_argument("-e", "--extension", type=str, default="jpg",
									help="specify the extension (default=.jpg)",)
parser.add_argument("-o", "--output", type=str, default="Output",
									help="specify the output directory",)
parser.add_argument("input_dir",    help="Specify the input directory")
args = parser.parse_args()

# Process parsed arguments
if args.debug:
	runmode = 2  # Show debug messages and intermediary steps
	runmode_str = "DEBUG"
elif args.fast:
	runmode = 0  # Do not show any debug messages or intermediary steps
	runmode_str = "FAST"
else:
	runmode = 1  # Default behaviour: show intermediary steps, but no debug
	runmode_str = "NORMAL"

# Set I/O directory names
input_directory = args.input_dir
extension = "*" + args.extension
files_directory = os.path.join(os.path.abspath(input_directory), extension)
files = glob.glob(files_directory)

output_directory = args.output
ensure_directory(output_directory) # No need to remove the output folder first

# Handle empty directory
if (not files):
    print("ERROR: Directory " + input_directory + " is empty!")
    sys.exit()

################
##### Main #####
################

# Constants
line_padding = 35
window_width = 40
stepsize = 20

# Run the program
print ("")
print ("Processing files from directory:".ljust(line_padding) + input_directory)
print ("With extension:".ljust(line_padding) + extension)
print ("Output Directory:".ljust(line_padding) + output_directory)
print ("Runmode:".ljust(line_padding) + runmode_str)
print ("")
times_list = []
initial_time = time.time()
end_time = initial_time
for image_name in files:
    start_time = time.time()
    _, image = os.path.split(image_name)
    
    image_name_base = os.path.splitext(image)[0]
    output_image_directory = os.path.abspath(os.path.join(output_directory, image_name_base))

    remove_directory(output_image_directory)
    ensure_directory(output_image_directory)

    # Preprocess image
    print ("")
    print ("Processing " + image)
    preprocessed_image = preprocess(image_name, output_image_directory, runmode=runmode)   
    print ("Finished preprocessing " + image)

    print ("    ****    ")
    
    # Segment preprocessed image
    print ("Segmenting " + image)    
    words_li_li = segment(preprocessed_image, output_image_directory, runmode=runmode)
    print ("Finished segmenting " + image)

    print ("    ****    ")

    # Classify segmented image
    print ("Classifying " + image)
    output_path = classify(output_image_directory, window_width=window_width, stepsize=stepsize, runmode=runmode)
    print ("Finished classifying " + image)
    print ("")
    print ("Output is in: " + output_path)
    print ("")

    # Duration of the whole pipeline for each image
    end_time = time.time()
    elapsed_time = end_time - start_time
    print ("Elapsed time: " + str(elapsed_time))
    print ("-" * line_padding)

    times_list.append(elapsed_time)

# Show statistics including total and average duration
total_elapsed_time = end_time - initial_time
average_time = np.average(times_list)
print ("Total elapsed time: " + str(total_elapsed_time))
print ("Average/image = " + str(average_time))
print ("-" * line_padding)
