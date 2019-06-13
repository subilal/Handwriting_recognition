from Preprocess.preprocess import *

'''
This file is intended only for runnning and testing
The preprocessing step.
'''

image_name = "P168-Fg016-R-C01-R01-fused.jpg"

input_directory = "TestInputPreprocess"
output_directory = "TestOutputPreprocess"
output_directory_for_segmentation = "TestInputSegment"
runmode = 2 # debug mode

input_image_name = os.path.join(input_directory, image_name)
print ("")
print ("Processing image " + image_name)
print ("")

remove_directory(output_directory)
ensure_directory(output_directory)
preprocessed_image = preprocess(input_image_name, output_directory, runmode=runmode)

write_image(preprocessed_image, output_directory_for_segmentation + "/image.jpg", runmode=runmode)

print ("")
print ("Finished Preprocessing image " + image_name)
print ("--------------------------------------------")
