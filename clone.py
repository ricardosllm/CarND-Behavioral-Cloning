import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
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
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    mesurement = float(line[3])
    mesurements.append(mesurement)

X_train = np.array(images)
y_train = np.array(mesurements)

model = Sequential()
# Normalize the data
model.add(Lambda(lambda x: x / 255.0 - 0.5,
                 input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train,
          validation_split=0.2,
          shuffle=True,
          epochs=7)

model.save('model.h5')
