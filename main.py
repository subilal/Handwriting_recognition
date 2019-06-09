import sys, os

import argparse
import glob

from Preprocess.preprocess import *
from Segment.segment import *

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
parser.add_argument("input_dir",    help="Specify the input directory")
args = parser.parse_args()

## Process parsed arguments
#if args.debug:
#    runmode = 2  # Show debug messages and intermediary steps
#elif args.fast:
#    runmode = 0  # Do not show any debug messages or intermediary steps
#else:
#    runmode = 1  # Default behaviour: show intermediary steps, but no debug
debug = args.debug

print ("extension=",args.extension)

#print ("runmode=" + str(runmode)) 

# Set I/O directory names
input_directory = os.path.abspath(args.input_dir)
extension = "*" + args.extension
files_directory = os.path.join(input_directory, extension)
files = glob.glob(files_directory)

output_directory = "Output" # This is a hardcoded value, but can easily be integrated in the command line args.
remove_directory(output_directory)
ensure_directory(output_directory)

################
##### Main #####
################

# Run the program
print ("")
print ("Processing files from the folder:")
print ("    " + input_directory)
print ("with extension:")
print ("    " + extension)
print ("")
for image_name in files:
    _, image = os.path.split(image_name)
    
    print ("")
    print ("Processing image " + image)

    image_name_base = os.path.splitext(image)[0]
    output_image_directory = os.path.abspath(os.path.join(output_directory, image_name_base))

    remove_directory(output_image_directory)
    ensure_directory(output_image_directory)

    rot_image, rot_line_peaks, rot_degree = preprocess(image_name, output_image_directory, debug=debug)

    write_image(rot_image, output_image_directory+"/OptimumRotation="+str(rot_degree)+".jpg", debug)
    
    print ("Finished preprocessing " + image)
    print ("    ****    ")
    print ("Segmenting image " + image)
    
    segment(rot_image, rot_line_peaks, output_image_directory, debug=debug)

    print ("Finished Segmenting image " + image)
    print ("--------------------------------------------")
