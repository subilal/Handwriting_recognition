import sys, os

import argparse
import glob

from Preprocess.preprocess import *
from Segment.segment import *



output_directory = "Output"

#parser = argparse.ArgumentParser()
#parser.add_argument('--d', help='run program in debug mode')
debug = False

#print (input_directory_argument)
#input_directory = glob.glob(input_directory_argument)



input_directory = "Input"
files = glob.glob("Input/*fused.jpg")
for image_name in files:
    _, image = os.path.split(image_name)
    
    print ("")
    print ("Processing image " + image)

    image_name_base = os.path.splitext(image)[0]
    output_directory = os.path.abspath(os.path.join(output_directory, image_name_base))

    remove_directory(output_directory)
    ensure_directory(output_directory)

    rot_image, rot_line_peaks, rot_degree = preprocess(image_name, output_directory, debug=debug)

    write_image(rot_image, output_directory+"/OptimumRotation="+str(rot_degree)+".jpg", debug)
    
    print ("Finished preprocessing " + image)
    print ("    ****    ")
    print ("Segmenting image " + image)
    
    segment(rot_image, rot_line_peaks, output_directory, debug=debug)

    print ("Finished Segmenting image " + image)
    print ("--------------------------------------------")
    break
