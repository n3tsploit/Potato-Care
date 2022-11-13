# Dataset gotten from https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/tywbtsjrjv-1.zip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import keras
from keras_preprocessing.image import img_to_array, ImageDataGenerator, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions


# Load our image to data generator

train_data = ImageDataGenerator(zoom_range=0.5, shear_range=0.3, rescale=1/255, horizontal_flip=True, preprocessing_function=)

validate_data = ImageDataGenerator(preprocessing_function=)

# load our image
train = train_data.flow_from_directory(directory='/content/Plant_leave_diseases_dataset_with_augmentation', target_size=(256,256), batch_size=32)
val = train_data.flow_from_directory(directory='/content/Plant_leave_diseases_dataset_with_augmentation', target_size=(256,256), batch_size=32)