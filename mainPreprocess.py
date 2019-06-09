from Preprocess.preprocess import *

'''
This file is intended only for runnning and testing
The preprocessing step.
'''

image_name = "P168-Fg016-R-C01-R01-fused.jpg"

input_directory = "TestInputPreprocess"
output_directory = "TestOutputPreprocess"
output_directory_for_segmentation = "TestInputSegment"
debug = True

input_image_name = os.path.join(input_directory, image_name)
print ("")
print ("Processing image " + image_name)
print ("")

remove_directory(output_directory)
ensure_directory(output_directory)
rot_image, rot_line_peaks, rot_degree = preprocess(input_image_name, output_directory, debug=debug)

print ("Optimum rotation=" + str(rot_degree))
write_image(rot_image, output_directory + "/optimumRotation=" + str(rot_degree) + ".jpg", debug=debug)
write_line_peaks(rot_line_peaks, output_directory+'/line_peaks')

write_image(rot_image, output_directory_for_segmentation + "/image.jpg", debug=debug)
write_line_peaks(rot_line_peaks, output_directory_for_segmentation + "/line_peaks")

print ("")
print ("Finished Preprocessing image " + image_name)
print ("--------------------------------------------")
