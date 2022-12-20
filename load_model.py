import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence