from Classifier.classifier import *

'''
This file is intended only for runnning and testing
The Classification step.
'''
image_name = "P168-Fg016-R-C01-R01-fused"

main_directory = "TestOutputclassify"
window_width=40
stepsize=20

io_main_dir = os.path.join(main_directory, image_name)

print ("Classifying " + image_name)
output_path = classify(io_main_dir, window_width, stepsize, 1)
print ("Finished classifying " + image_name)
print ("")
print ("Output is in: " + output_path)
print ("")