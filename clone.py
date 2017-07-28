import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
mesurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        mesurement = float(line[3])
        mesurements.append(mesurement)

augmented_images, augmented_mesurements = [], []
for image, mesurement in zip(images, mesurements):
    augmented_images.append(image)
    augmented_mesurements.append(mesurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_mesurements.append(mesurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_mesurements)

model = Sequential()
# Normalize the data
model.add(Lambda(lambda x: x / 255.0 - 0.5,
                 input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# NVIDIA architecture
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# LeNet
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train,
          validation_split=0.2,
          shuffle=True,
          epochs=5)

model.save('model.h5')
