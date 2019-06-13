import tensorflow.keras as keras
import idx2numpy
import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
from skimage import io
import pickle
import transcription

with open('./Classifier/LabelEncoder.pickle', 'rb') as f:
	LabelEncoder = pickle.load(f)
with open('./Classifier/OneHotEncoder.pickle', 'rb') as f:
	OneHotEncoder = pickle.load(f)
model = keras.models.load_model('./Classifier/trained_cnn.h5')
stepsize = 10
width_threshold = 12

def classify(output_directory, runmode=1):

	alpha_name = "alpha.txt"
	alpha_txt = open(output_directory + "/" + alpha_name,'w')
	lines_directory = output_directory + '/lines'

	line_labels = [ line for line in os.listdir(lines_directory) if os.path.isdir(os.path.join(lines_directory, line)) ]
	line_labels.sort()
	for line_label in line_labels:
		# print(line_label)
		line_path = os.path.join(lines_directory, line_label)
		word_labels = os.listdir(line_path)
		word_labels.sort()
		for word_label in word_labels:
			# print(word_label)
			word_path = os.path.join(line_path, word_label)
			word_image = cv.imread(word_path)
			
			# cv.imshow("", word_image)
			print(word_image.shape)
			# cv.waitKey(0)
			
			height = word_image.shape[0]
			width = 40 if word_image.shape[1] > 40 else word_image.shape[1]
			letter_images = sliding_window(word_image, [height, width], stepsize)
			
			letters = predict(letter_images)
			letters.reverse()
			for l in letters:
				text = text + '-' + l

			text = text + " "

		alpha_txt.write(text)
		alpha_txt.write("\n")
	alpha_txt.close()
	return transcription.transcript(output_directory, alpha_name)

	# Idea: return the string.
	# For runmode > 1 (i.e. debug mode), we also want to save the .txt file

def sliding_window(image, windowSize, stepSize=10):
	letter_images = []
	flag = 0
	# slide a window across the image
	for x in range(0, image.shape[1], stepSize):
		# yield the current window
		if x + windowSize[1] < image.shape[1]:
			window_img = image[0:windowSize[0], x:x + windowSize[1]]
		else:
			window_img = image[0:windowSize[0], x:image.shape[1]]
			flag = 1

		re_image = cv.resize(window_img, (35,45))
		# cv.imshow("", re_image)
		# print(re_image.shape)
		# cv.waitKey(0)
		letter_images.append(re_image)
		
		if flag == 1:
			break
	print(len(letter_images))
	return np.array(letter_images)

def predict(images):
	letters = []
	predicted = model.predict(images, batch_size=images.shape[0])
	print(predicted)
	for prob_values in predicted:
		if max(prob_values) > 80:
			print(np.argmax(prob_values))
			letter = LabelEncoder.inverse_transform([np.argmax(prob_values)])
			letters.append(letter)
	return letters

if __name__ == "__main__":
	classify = classify("./TestOutputSegment")







