from Preprocess.preprocess import *

'''
This file is intended only for runnning and testing
The preprocessing step.
'''

input_image_name = "TestInputPreprocess/image.jpg"
output_directory = "TestOutputPreprocess"
output_directory_for_segmentation = "TestInputSegment"

rot_image, rot_line_peaks, rot_degree = preprocess(input_image_name, output_directory, debug=True)

print ("Optimum rotation=" + str(rot_degree))
print ("Optimum line peaks=\n" + str(rot_line_peaks))
write_image(rot_image, output_directory+'/optimumRotation.jpg')
write_line_peaks(rot_line_peaks, output_directory+'/line_peaks')

write_image(rot_image, output_directory_for_segmentation+'/image.jpg')
write_line_peaks(rot_line_peaks, output_directory_for_segmentation+'/line_peaks')

print ("Main Preprocessing successful")
