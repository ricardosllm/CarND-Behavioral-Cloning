import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from sklearn import model_selection
from data import generate_samples, preprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

local_project_path = './'
local_data_path    = os.path.join(local_project_path, 'data')
batch_size         = 128

dlog_csv_path = os.path.join(local_data_path, 'driving_log.csv')
driving_log   = pd.io.parsers.read_csv(dlog_csv_path)
train, valid  = model_selection.train_test_split(driving_log, test_size=.2)


model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(32, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.compile(optimizer=Adam(lr=1e-04), loss='mean_squared_error')

history = model.fit_generator(
    generate_samples(train, local_data_path, batch_size),
    steps_per_epoch=train.shape[0] / batch_size,
    nb_epoch=1,
    validation_data=generate_samples(valid, local_data_path, batch_size),
    validation_steps=valid.shape[0] / batch_size
)

model.save('model.h5')
