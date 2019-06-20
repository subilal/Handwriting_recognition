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
		self.network.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(45,35,1)))
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
				grayImage = cv.cvtColor(re_image, cv.COLOR_BGR2GRAY)
				(thresh, BW) = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY)
				data.append(BW)
				data_labels.append(label)
		
		data = np.array(data, dtype="float") / 255.0
		data = data.reshape(data.shape[0], 45, 35, 1)
		data_labels = np.array(data_labels)
		return [data, data_labels]


if __name__ == "__main__":
	data_path = 'Train-input'
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
	onehot_encoder = OneHotEncoder(sparse=False)
	
	le = label_encoder.fit(labels)
	integer_encoded = le.transform(labels)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

	ohe = onehot_encoder.fit(integer_encoded)

	(train_images, test_images, train_labels, test_labels) = train_test_split(data,	integer_encoded, test_size=0.25, random_state=42)

	datagen = keras.preprocessing.image.ImageDataGenerator(
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=False,
			vertical_flip=False,
			fill_mode='nearest')

	datagen.fit(train_images)

	# cv.imshow("", train_images[10])
	# print(train_images[10].shape)
	# cv.waitKey(0)

	# train_labels = keras.utils.to_categorical(train_labels)
	# test_labels = keras.utils.to_categorical(test_labels)
	train_labels = ohe.transform(train_labels)
	test_labels = ohe.transform(test_labels)
	opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.network.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	hist = model.network.fit_generator(datagen.flow(train_images, train_labels, batch_size=100), validation_data=(test_images, test_labels), epochs=100)
	model.network.save('trained_cnn.h5')

	with open('trainHistoryDict.pickle', 'wb') as file:
		pickle.dump(hist.history, file, pickle.HIGHEST_PROTOCOL)
	with open('LabelEncoder.pickle', 'wb') as file:
		pickle.dump(le, file, pickle.HIGHEST_PROTOCOL)
	with open('OneHotEncoder.pickle', 'wb') as file:
		pickle.dump(ohe, file, pickle.HIGHEST_PROTOCOL)
