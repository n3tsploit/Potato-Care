# load best model
import numpy as np
from keras.applications.convnext import preprocess_input

from keras.models import load_model
from keras.utils import load_img, img_to_array

model = load_model('/content/PlantDNet.h5')


# acc = model.evaluate_generator(val)[1]
#
# print(f'The accuracy of the model is {acc*100}%')

def prediction(path):
    img = load_img(path, target_size=(64, 64,3))

    i = img_to_array(img)
    im = preprocess_input(i)

    img = np.expand_dims(im, axis=0)

    pred = np.argmax(model.predict(img))

    print(pred)


path = '/content/PlantVillage/Potato___Early_blight/044c3abc-0bc9-45fb-8fd5-094aeb605f90___RS_Early.B 8044.JPG'

prediction(path)
train.class_indices
