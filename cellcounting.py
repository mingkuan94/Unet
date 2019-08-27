#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:46:29 2019

@author: mingkuan
"""

from PIL import Image
import os
import numpy as np
import csv

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

### read training labels ####
os.chdir("/home/mingkuan/Desktop/Data Files_Question1_SSC2019CaseStudy")#please change the directory to your working path
training_image_name = []
training_cell_count = []
training_blur_level = []
training_stain = []
with open('train_label.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader,None)
    for row in reader:
        training_image_name.append(row[0])
        training_cell_count.append(row[1])
        training_blur_level.append(row[2])
        training_stain.append(row[3])

### read training images ####
os.chdir("/home/mingkuan/Desktop/Data Files_Question1_SSC2019CaseStudy/train/")#please change the directory to your working path
##create a list (training_list) containing pixel value of 2400 training images
traning_list = []
for i in training_image_name:
	im = Image.open(i)
	#im.show()
	imarray = np.array(im)
	traning_list.append(imarray)

training_list = traning_list
training_list = np.array(training_list)
training_list.shape



### read test labels ####
os.chdir("/home/mingkuan/Desktop/Data Files_Question1_SSC2019CaseStudy")#please change the directory to your working path
testing_image_name = []
testing_cell_count = []
testing_blur_level = []
testing_stain = []
with open('test_label.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader,None)
    for row in reader:
        testing_image_name.append(row[0])
        #testing_cell_count.append(row[1])
        testing_blur_level.append(row[1])
        testing_stain.append(row[2])

## read testing images ####        
os.chdir("/home/mingkuan/Desktop/Data Files_Question1_SSC2019CaseStudy/test/")#please change the directory to your working path
##create a list (testing_list) containing pixel value of 1200 testing images
testing_list = []
for i in testing_image_name:
	im = Image.open(i)
	#im.show()
	imarray = np.array(im)
	testing_list.append(imarray)

testing_list = np.array(testing_list)
testing_list.shape


os.chdir('/home/mingkuan/Desktop/')
train_x_resize = np.load('train_x_resize.npy')
train_x_mask = np.load('train_x_mask.npy')
test_x_resize = np.load('test_x_resize.npy')
test_x_mask = np.load('test_x_mask.npy')

training_cell_count = np.array(training_cell_count)
training_cell_count = training_cell_count.reshape((2400,1))
training_cell_count.shape

train_x_mask.shape






img_rows=256
img_cols=256
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#CNN Model
# Model: 11 Weighted Layers
model = Sequential()

# CNN-1: 3-64x2
model.add(Conv2D(nb_filters, kernel_size=(3,3),
                        padding='same',
                        input_shape=(img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters, kernel_size=(3,3),
                        padding='same',
                        input_shape=(img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((nb_pool, nb_pool)))

# CNN-2: 3-128x2
model.add(Conv2D(nb_filters*2, kernel_size=(3,3),
                        padding='same',
                        input_shape=( img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters*2, kernel_size=(3,3),
                        padding='same',
                        input_shape=( img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((nb_pool, nb_pool)))

# CNN-3: 3-256x2
model.add(Conv2D(nb_filters*4,kernel_size=(3,3),
                        padding='same',
                        input_shape=( img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters*4, kernel_size=(3,3),
                        padding='same',
                        input_shape=(img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((nb_pool, nb_pool)))

# # CNN-4: 3-512x2
model.add(Conv2D(nb_filters*8, kernel_size=(3,3),
                        padding='same',
                        input_shape=( img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters*8, kernel_size=(3,3),
                        padding='same',
                        input_shape=( img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((nb_pool, nb_pool)))

# FC-1024x2 Fully connected layers
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# FC-30 Last layer
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')#, metrics=[mean_iou])
model.summary()


earlystopper = EarlyStopping(patience=5, verbose=1)
#checkpointer = ModelCheckpoint('model-ssc2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(train_x_mask, training_cell_count, validation_split=0.3, batch_size=8, epochs=2, callbacks=[earlystopper])

total_area = []
for i in range(len(train_x_mask)):
    im = train_x_mask[i].reshape((256*256,))
    total = sum(im)
    total_area.append(total)

len(total_area)
total_area = np.array(total_area)
