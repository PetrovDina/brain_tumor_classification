from sklearn.metrics import  classification_report,confusion_matrix
from tensorflow import keras

import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

model = keras.models.load_model('model.h5')

dataPath = "./data"
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Setting hyper-parameters
image_height = 224
image_width = 224

# Importing images
data_set_images = []
data_set_labels = []
for label in labels:
    pathToFolder = os.path.join(dataPath, 'Training', label)
    for file in os.listdir(pathToFolder):
        img = cv2.imread(os.path.join(pathToFolder, file))
        img = cv2.resize(img, (image_height, image_width))
        data_set_images.append(img)
        data_set_labels.append(label)

print("Finished importing training images")

for label in labels:
    pathToFolder = os.path.join(dataPath, 'Testing', label)
    for file in os.listdir(pathToFolder):
        img = cv2.imread(os.path.join(pathToFolder, file))
        img = cv2.resize(img, (image_height, image_width))
        data_set_images.append(img)
        data_set_labels.append(label)

print("Finished importing test images")

data_set_images = np.array(data_set_images)
data_set_labels = np.array(data_set_labels)

print(data_set_images.shape)

# Shuffling data
data_set_images, data_set_labels = shuffle(data_set_images, data_set_labels, random_state=101)

# Image data augmentation
datagen = ImageDataGenerator(   # TODO dodati i promijeniti neke parametre!
    rotation_range=30,
    width_shift_range=0.1,
    rescale=1./255,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True)

datagen.fit(data_set_images)


# Creating training and testing sets
train_ratio = 0.70
validation_ratio = 0.20
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(data_set_images, data_set_labels,
                                                    test_size=test_ratio, random_state=101)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
y_test_new = np.argmax(y_test, axis=1)
print(classification_report(y_test_new, pred))


#matrica knf
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
sns.heatmap(confusion_matrix(y_true=y_test_new, y_pred=pred), ax=ax, xticklabels=labels, yticklabels=labels, annot=True,
           alpha=0.7, linewidths=2)
fig.text(s='Heatmap of the Confusion Matrix', size=18, fontweight='bold',
             fontname='monospace', y=0.92, x=0.28, alpha=0.8)

plt.show()
