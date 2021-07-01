

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import backend as K, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.utils import shuffle #ovo isto? vrv TODO
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

#---1.7.2021
from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
#------


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
# Shuffling (TODO  use kaggle shuffle) TODO MOZDA NE RADI
#np.random.shuffle(data_set_images) izmjeniti u buducnosti

# Image data augmentation here TODO ovde ide i deljenje sa 255 ILI NE?

data_set_images, data_set_labels = shuffle(data_set_images, data_set_labels, random_state=101)
#----------------------------------- dodato 1.7.2021
datagen = ImageDataGenerator(   #dodati i promijeniti neke parametre!
    rotation_range=30,
    width_shift_range=0.1,
    rescale=1./255,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True)

datagen.fit(data_set_images)
#-

#--------------------------
print(data_set_images.shape)

# Creating training and testing sets
#train_ratio = 0.70
#validation_ratio = 0.20
#test_ratio = 0.10

#x_train, x_test, y_train, y_test = train_test_split(data_set_images, data_set_labels, test_size=1 - train_ratio)
#x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

x_train, x_test, y_train, y_test = train_test_split(data_set_images, data_set_labels, test_size=0.1,random_state=101)
#---------- 1.7.2021

y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

#y_val_new = []
#for i in y_val:
#    y_val_new.append(labels.index(i))
#y_val = y_val_new
#y_val = tf.keras.utils.to_categorical(y_val)

#----

# Transfer learning
effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))

model = effnet.output
model = GlobalAveragePooling2D()(model)
model = Dropout(rate=0.5)(model)
model = Dense(4, activation='softmax')(model)
model = Model(inputs=effnet.input, outputs=model)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint("effnet.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001,
                              mode='auto', verbose=1)

history = model.fit(x_train,y_train,validation_split=0.1, epochs =12, verbose=1, batch_size=32,
                   callbacks=[tensorboard,checkpoint,reduce_lr]) #TODO: izmjeniti validaciju da bude 0.2
# history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                              epochs=epochs, validation_data=(x_val, y_val),
#                              steps_per_epoch=x_train.shape[0] // batch_size,
#                              callbacks=[tensorboard, checkpoint, reduce_lr])

# model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size) #fir sa stack overflow

# history = model.fit(x_train,y_train,validation_split=0.1, epochs =12, verbose=1, batch_size=32)
#                    #callbacks=[tensorboard,checkpoint,reduce_lr]) #fit sa primarnog lika


# Loss graph
plt.plot(history.history["loss"],c = "purple")
plt.plot(history.history["val_loss"],c = "orange")
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["train", "validation"])
plt.show()

# Accuracy graph
plt.plot(history.history["accuracy"],c = "purple")
plt.plot(history.history["val_accuracy"],c = "orange")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["train", "validation"])
plt.show()