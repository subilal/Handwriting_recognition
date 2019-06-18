from Segment.segment import *

'''
This file is intended only for runnning and testing
The segmentation step.
'''

image_name = "image.jpg"

input_directory = "TestInputSegment"
output_directory = "TestOutputSegment"

input_image_name = os.path.join(input_directory, image_name)

image = read_image (input_image_name)

print ("")
print ("Segmenting image " + image_name)
print ("")

words_li_li = segment(image, output_directory, runmode=2)

print ("")
print ("Finished Segmenting image " + image_name)
print ("--------------------------------------------")

number_lines = len(words_li_li)
line_count = 0
print ("There are " + str(number_lines) + " lines, each having:")
for words in words_li_li:
    line_count = line_count + 1
    number_words = len(words)
    print ("Line " + str(line_count) + ": " + str(number_words) + " words")
