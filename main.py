# !wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/tywbtsjrjv-1.zip
# !unzip tywbtsjrjv-1.zip
# !unzip Plant_leaf_diseases_dataset_with_augmentation.zip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import keras
from keras_preprocessing.image import img_to_array, ImageDataGenerator, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

# PREPROCESSING THE IMAGE

# Load our image to data generator

train_data = ImageDataGenerator(zoom_range=0.5, shear_range=0.3, rescale=1 / 255, horizontal_flip=True,
                                preprocessing_function=preprocess_input)

validate_data = ImageDataGenerator(preprocessing_function=preprocess_input)

# load our image
train = train_data.flow_from_directory(directory='/content/Plant_leave_diseases_dataset_with_augmentation',
                                       target_size=(256, 256), batch_size=32)
val = train_data.flow_from_directory(directory='/content/Plant_leave_diseases_dataset_with_augmentation',
                                     target_size=(256, 256), batch_size=32)

t_img, label = train.next()


def plot_image(img_arr, label):
    for im, l in zip(img_arr, label):
        plt.figure(figsize=(5, 5))
        plt.show()


# BUILDING OUR MODEL
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
import keras

base_model = VGG19(input_shape=(256, 256, 3), include_top=False)
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(units=38, activation='softmax')(x)

# creating our model
model = Model(base_model.input, x)

model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# EARLY STOPING AND CHECKPOINTS
from keras.callbacks import ModelCheckpoint, EarlyStopping

# early stopping
es = EarlyStopping(monitor= 'val_accuracy', min_delta=0.01, patience=3, verbose=1)

# model checkpoint
mc = ModelCheckpoint(filepath='bestmodel.h5',
                     monitor='val_accuracy', min_delta = 0.01,
                     patience = 3,
                     verborse = 1,
                     save_best_only= True)

cb = [es, mc]

his = model.fit_generator(train, steps_per_epoch=16, epochs=50, verbose = 1, callbacks= cb, validation_data=val, validation_steps=16)


