import os
import csv
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
from sklearn import model_selection
from data import generate_samples, preprocess


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class WeightsSaver(keras.callbacks.Callback):
    """
    Keeps track of model weights by saving them at the end of each epoch.
    """

    def __init__(self, root_path):
        super(WeightsLogger, self).__init__()
        self.weights_root_path = os.path.join(root_path, 'weights/')
        shutil.rmtree(self.weights_root_path, ignore_errors=True)
        os.makedirs(self.weights_root_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs={}):
        weigths_path = os.path.join(self.weights_root_path,
                                    'model_epoch_{}.h5'.format(epoch + 1))
        self.model.save_weights(weigths_path)

local_project_path = './'
local_data_path    = os.path.join(local_project_path, 'data')

driving_log_path = os.path.join(local_data_path, 'driving_log.csv')
driving_log      = pd.io.parsers.read_csv(driving_log_path)
train, valid     = model_selection.train_test_split(driving_log, test_size=.2)

model = models.Sequential()
model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
model.add(core.Dense(500, activation='relu'))
model.add(core.Dropout(.5))
model.add(core.Dense(100, activation='relu'))
model.add(core.Dropout(.25))
model.add(core.Dense(20, activation='relu'))
model.add(core.Dense(1))
model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')

history = model.fit_generator(
    generate_samples(train, local_data_path),
    samples_per_epoch=train.shape[0],
    nb_epoch=1,
    validation_data=generate_samples(valid, local_data_path),
    callbacks=[WeightsSaver(root_path=local_project_path)],
    nb_val_samples=valid.shape[0]
)

with open(os.path.join(local_project_path, 'model.json'), 'w') as file:
    file.write(model.to_json())

backend.clear_session()
