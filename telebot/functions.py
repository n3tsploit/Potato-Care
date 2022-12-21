import tensorflow as tf
import numpy as np


def predict(image_name):
    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

    model = tf.keras.models.load_model('potatoes.h5')
    # Load the image
    image = tf.io.read_file(image_name)

    # Decode the image
    image = tf.image.decode_image(image)

    # Preprocess the image
    image = tf.image.resize(image, (256, 256))  # Resize to model input size
    image = image / 255.0  # Normalize pixel values

    # Add an additional dimension to the input tensor
    image = tf.expand_dims(image, axis=0)

    # Make predictions
    predictions = model.predict(image)

    return class_names[np.argmax(predictions)]