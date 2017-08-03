import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn import model_selection
from data import generate_samples, preprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.switch_backend('agg')


local_project_path = './'
local_data_path    = os.path.join(local_project_path, 'data')

driving_log_path = os.path.join(local_data_path, 'driving_log.csv')
driving_log      = pd.io.parsers.read_csv(driving_log_path)
train, valid     = model_selection.train_test_split(driving_log, test_size=.2)


# NVIDIA architecture
model = Sequential()
model.add(Conv2D(24,5,5, input_shape=(32, 128, 3),
                 strides=(2,2), activation='relu'))
model.add(Conv2D(36,5,5, strides=(2,2), activation='relu'))
model.add(Conv2D(48,5,5, strides=(2,2), activation='relu'))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(
    generate_samples(train, local_data_path),
    samples_per_epoch=train.shape[0],
    nb_epoch=30,
    validation_data=generate_samples(valid, local_data_path),
    nb_val_samples=valid.shape[0]
)

model.save('model.h5')

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
