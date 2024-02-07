import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import keras

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)
train_dataset = train.flow_from_directory('basedata/train/', target_size=(200, 200), batch_size=32, class_mode='binary')
validation_dataset = validation.flow_from_directory('basedata/validation/', target_size=(200, 200), batch_size=32, class_mode='binary')

model = keras.models.Sequential([
    keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(16, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

model_fit = model.fit(train_dataset, steps_per_epoch=3, epochs=1000, validation_data=validation_dataset)
