
# Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import backend as K, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report, confusion_matrix
import seaborn as sns

# Path to dataset
dataPath = "./data"

# Labels (classes)
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Hyper-parameters
image_height = 224
image_width = 224
batch_size = 16 # treba 32!!
num_classes = len(labels)
epochs = 12


def import_training_images(data_set_images, data_set_labels):

    for label in labels:
        pathToFolder = os.path.join(dataPath, 'Training', label)
        for file in os.listdir(pathToFolder):
            img = cv2.imread(os.path.join(pathToFolder, file))
            img = cv2.resize(img, (image_height, image_width))
            data_set_images.append(img)
            data_set_labels.append(label)

    print("Finished importing training folder images")
    return data_set_images, data_set_labels


def import_test_images(data_set_images, data_set_labels):

    for label in labels:
        pathToFolder = os.path.join(dataPath, 'Testing', label)
        for file in os.listdir(pathToFolder):
            img = cv2.imread(os.path.join(pathToFolder, file))
            img = cv2.resize(img, (image_height, image_width))
            data_set_images.append(img)
            data_set_labels.append(label)

    print("Finished importing test folder images")
    return data_set_images, data_set_labels


def image_data_augmentation():
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        vertical_flip = True,
    )

    return datagen



def one_hot_encoding(y):
    y_new = []
    for i in y:
        y_new.append(labels.index(i))
    y = y_new
    y = tf.keras.utils.to_categorical(y)
    return y


def create_model():
    effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))

    model = effnet.output
    model = GlobalAveragePooling2D()(model)
    model = Dropout(rate=0.5)(model)
    model = Dense(4, activation='softmax')(model)
    model = Model(inputs=effnet.input, outputs=model)

    return model


def train(model):

    # Callbacks
    tensorboard = TensorBoard(log_dir='logs')
    checkpoint = ModelCheckpoint("model.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001,
                                  mode='auto', verbose=1)

    # Training
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, verbose=1, batch_size=batch_size,
                        callbacks=[tensorboard, checkpoint, reduce_lr])  # TODO: izmjeniti validaciju da bude 0.2

    return history



def plot_loss_graph(history):
    plt.plot(history.history["loss"], c="purple")
    plt.plot(history.history["val_loss"], c="orange")
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["train", "validation"])
    plt.show()


def plot_accuracy_graph(history):
    plt.plot(history.history["accuracy"], c="purple")
    plt.plot(history.history["val_accuracy"], c="orange")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(["train", "validation"])
    plt.show()


def plot_confusion_matrix(y_test_new, prediction):
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    sns.heatmap(confusion_matrix(y_true=y_test_new, y_pred=prediction), ax=ax, xticklabels=labels, yticklabels=labels,
                annot=True,
                alpha=0.7, linewidths=2)
    fig.text(s='Confusion Matrix', size=20, fontweight='bold',
             fontname='monospace', y=0.92, x=0.28, alpha=0.8)

    plt.show()


if __name__ == '__main__':

    data_set_images = []  # Images
    data_set_labels = []  # Image labels

    # Importing training folder images
    data_set_images, data_set_labels = import_training_images(data_set_images, data_set_labels)

    # Importing test folder images
    data_set_images, data_set_labels = import_test_images(data_set_images, data_set_labels)

    data_set_images = np.array(data_set_images)
    data_set_labels = np.array(data_set_labels)

    print(data_set_images.shape)

    # Shuffling data
    data_set_images, data_set_labels = shuffle(data_set_images, data_set_labels, random_state=101)

    # Image data augmentation
    datagen = image_data_augmentation()
    datagen.fit(data_set_images)

    # Creating training and testing sets
    train_ratio = 0.70
    validation_ratio = 0.20
    test_ratio = 0.10

    x_train, x_test, y_train, y_test = train_test_split(data_set_images, data_set_labels,
                                                        test_size=test_ratio, random_state=101)

    # One hot encoding
    y_train = one_hot_encoding(y_train)
    y_test = one_hot_encoding(y_test)

    # Model
    model = create_model()
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # Training
    history = train(model)

    # Loss and accuracy plots
    plot_loss_graph(history)
    plot_accuracy_graph(history)

    # Prediction
    prediction = model.predict(x_test)
    prediction = np.argmax(prediction, axis=1)
    y_test_new = np.argmax(y_test, axis=1)
    print(classification_report(y_test_new, prediction))

    # Evaluation
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Confusion matrix
    plot_confusion_matrix(y_test_new, prediction)



