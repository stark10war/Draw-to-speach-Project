# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 13:22:09 2019

@author: Shashank
"""



import os
os.chdir('D:/Python practice/Hand Gesture')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import _pickle as pickel
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing import image

PATH = os.getcwd()
image_dir =  os.path.join(PATH, "Train_images")

character_labels = []
text_images = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root,file)
        label = os.path.basename(root)
        print(label,path)
        img = image.load_img(path, target_size=(28,28), color_mode='grayscale')        
        img = image.img_to_array(img)
        character_labels.append(label)
        text_images.append(img)


train = np.array(text_images)

train = train/255



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
character_labels = labelencoder.fit_transform(character_labels)
pickel.dump(labelencoder,  open('labelencoder.dat', 'wb'))


labels_onehot = keras.utils.to_categorical(character_labels, num_classes=27)

from sklearn.model_selection import train_test_split 

xtrain, xtest, ytrain, ytest = train_test_split(train,labels_onehot, test_size = 0.3, random_state = 24)


# Set the CNN Architecture
input_shape = (28,28,1)
num_classes  = 27

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Comple the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.summary()

import random
random.seed(24)
# Train the model
model.fit(xtrain, ytrain, batch_size=200, epochs=20, verbose=1, validation_data=(xtest, ytest),shuffle= True)


model.save('text_recognizer.h5')







