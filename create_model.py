import numpy as np
import matplotlib.pyplot as plt
import keras
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

train_datagen = ImageDataGenerator(zoom_range=0.5, shear_range=0.3, rescale=1 / 255, horizontal_flip=True,
                                   preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

data_base_path = "/content/PlantVillage"
train = train_datagen.flow_from_directory(directory=data_base_path, target_size=(256, 256), batch_size=32)
val = val_datagen.flow_from_directory(directory=data_base_path, target_size=(256, 256), batch_size=32)

# Creating model
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19

base_model = VGG19(input_shape=(256, 256, 3), include_top=False)
for label in base_model.layers:
    label.trainable = False

base_model.summary()
x = Flatten()(base_model.output)
x = Dense(units=38, activation='softmax')(x)
model = Model(base_model.input, x)
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping

# eary stoping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1)
# model check point
mc = ModelCheckpoint(filepath="best_model.h5", monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1,
                     save_best_only=True)
cb = [es, mc]
his = model.fit_generator(train, steps_per_epoch=16, epochs=50, verbose=1, callbacks=cb, validation_data=val,
                          validation_steps=16)

print('Model created....')