#Refer - https://www.youtube.com/watch?v=5JAZiue-fzY

import os
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing import image

d = {
    0: 'Black Sea Sprat',
    1: 'Gilt-Head Bream',
    2: 'Hourse Mackerel',
    3: 'Red Mullet',
    4: 'Red Sea Bream',
    5: 'Sea Bass',
    6: 'Shrimp',
    7: 'Striped Red Mullet',
    8: 'Trout'
}

# print(tf.version.VERSION)

fmodel = tf.keras.models.load_model('best_fish_model.h5')

def prepare_img(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

img = prepare_img('test images/Red Mullet.png')

results = fmodel.predict(img)
print(results)
print(results.shape)
print(d[list(results[0]).index(max(results[0]))])


#========REFERENCES==========
# https://stackoverflow.com/questions/69174223/valueerror-input-0-is-incompatible-with-layer-model-2-expected-shape-none-16