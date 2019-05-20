# Only for training the network

import tensorflow.keras as keras
import idx2numpy
import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
from skimage import io
import gzip
import pickle
import matplotlib.pyplot as plt

class CNN():

	def __init__(self):
		self.network = keras.Sequential()
		self.network.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(45,35,3)))
		self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
		self.network.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
		self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
		self.network.add(keras.layers.Flatten())
		self.network.add(keras.layers.Dense(27, activation='softmax'))

	def extract_data(self, filepath):
		data = []
		data_labels = []

		for label in os.listdir(filepath):
			label_path = os.path.join(filepath, label)
			count = 0
			list = len(os.listdir(label_path))
			for img in os.listdir(label_path):
				image_path = os.path.join(label_path, img)
				image = cv.imread(image_path)
				re_image = cv.resize(image, (35,45))
				data.append(re_image)
				data_labels.append(label)
		
		data = np.array(data, dtype="float") / 255.0
		data = data.reshape(data.shape[0], 45, 35, 3)
		data_labels = np.array(data_labels)
		return [data, data_labels]


if __name__ == "__main__":

	data_path = '../character_data'
	model = CNN()	

	[data, labels] = model.extract_data(data_path)

# show image using cv
	# cv.imshow("", train_data[512])
	# cv.waitKey(0)

# show image using matplotlib
	# image = np.asarray(train_data[512]).squeeze()
	# plt.imshow(image)
	# plt.show()
	
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(labels)	
	(train_images, test_images, train_labels, test_labels) = train_test_split(data,	integer_encoded, test_size=0.25, random_state=42)

	# cv.imshow("", train_images[10])
	# cv.waitKey(0)

	train_labels = keras.utils.to_categorical(train_labels)
	test_labels = keras.utils.to_categorical(test_labels)
	opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.network.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	hist = model.network.fit(train_images, train_labels, batch_size = 15, validation_data=(test_images, test_labels), epochs=35)
	model.network.save('trained_cnn.h5')
	with open('trainHistoryDict', 'wb') as file:
		pickle.dump(hist.history, file)
