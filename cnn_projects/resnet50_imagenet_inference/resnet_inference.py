#Keras - Pre-Trained Models
import keras
import tensorflow as tf
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
#from keras.applications.resnet import re
from keras.applications import resnet
#filename = 'banana.jpg'
filename='orange.jpg'
original =tf.keras.preprocessing.image.load_img(filename,target_size = (224,224))
print('PIL image size',original.size)
plt.imshow(original)
plt.show()
#convert the PIL image to a numpy array
numpy_image = tf.keras.preprocessing.image.img_to_array(original)
plt.imshow(np.uint8(numpy_image))
plt.show()
print('numpy array size',numpy_image.shape)
# Convert the image / images into batch format
image_batch = np.expand_dims(numpy_image, axis = 0)
print('image batch size', image_batch.shape)
processed_image = resnet.preprocess_input(image_batch.copy())
# create resnet model
resnet_model = resnet.ResNet50(weights = 'imagenet')
# get the predicted probabilities for each class
predictions = resnet_model.predict(processed_image)
# convert the probabilities to class labels
label = decode_predictions(predictions)
print(label)