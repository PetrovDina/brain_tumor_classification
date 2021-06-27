
# importi sa vezbi
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D # global dodat
from keras import backend as K


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
#from tqdm import tqdm #ovo nam vrv ne treba za ispis na konzolu
import os
#from sklearn.utils import shuffle #ovo isto? vrv
from sklearn.model_selection import train_test_split

#ovo nam vrv ne treba
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import io

#----------------------------
#Pitanje 1: da li da koristimo sequential

dataPath = "./data"
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']


# Setting hyper-parameters
image_height = 224 # ili 150??
image_width = 224
batch_size = 32
num_classes = len(labels)
epochs = 12 # ili 100??


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

# 3264 slike ukupno,
print(data_set_images.shape)
# Shuffling (TODO maybe use kaggle shuffle) TODO MOZDA NE RADI
np.random.shuffle(data_set_images)

# Image data augmentation here TODO ovde ide i deljenje sa 255
#datagen_val = ImageDataGenerator(rescale=1./255)


# Creating training and testing sets
train_ratio = 0.70
validation_ratio = 0.20
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(data_set_images, data_set_labels, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

