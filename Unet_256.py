#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 00:54:34 2019

@author: mingkuanwu
"""

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

from PIL import Image

# Set some parameters
IMG_WIDTH = 128*2
IMG_HEIGHT = 128*2
IMG_CHANNELS = 1

#TRAIN_PATH = '/Users/mingkuanwu/Desktop/unet/stage1_train/'
#TEST_PATH = '/Users/mingkuanwu/Desktop/unet/stage1_test/'

#TRAIN_PATH = '/Users/mingkuanwu/Desktop/unet/BBBC005_v1_images/'
#Y_PATH = '/Users/mingkuanwu/Desktop/unet/BBBC005_v1_ground_truth/synthetic_2_ground_truth/'

# Ubuntu path
TRAIN_X_PATH = '/home/mingkuan/Desktop/BBBC005_v1_images/'
TRAIN_Y_PATH = '/home/mingkuan/Desktop/BBBC005_v1_ground_truth/synthetic_2_ground_truth/'


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed



# Get train and test IDs
#train_ids = next(os.walk(TRAIN_PATH))[1]
#test_ids = next(os.walk(TEST_PATH))[1]


y_train_ids = []
for _, _, files in os.walk(TRAIN_Y_PATH):  
    for filename in files:
        y_train_ids.append(filename)
len(y_train_ids)

for i in range(len(y_train_ids)-1):
    if y_train_ids[i] == '.htaccess':
        del y_train_ids[i]
len(y_train_ids)        




#len(test_ids)

# Get and resize train images and masks

Y_train = np.zeros((len(y_train_ids), IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.bool)
Y_train.shape
print('Getting train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(y_train_ids), total=len(y_train_ids)):
    path = TRAIN_Y_PATH + id_
    img = Image.open(path)
    ncols, nrows = img.size
    img = np.array(img.getdata()).reshape((nrows, ncols,1))
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True)
    #img = imread(path)[:,:]
    #img = img.reshape((520,696,1))
    #img = resize(img, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True)
    Y_train[n] = img

    
X_train = np.zeros((len(y_train_ids), IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.uint8)    
for n, id_ in tqdm(enumerate(y_train_ids), total=len(y_train_ids)):
    path = TRAIN_X_PATH + id_
    img = Image.open(path)
    ncols, nrows = img.size
    img = np.array(img.getdata()).reshape((nrows, ncols,1))
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True)
    #img = imread(path)[:,:]
    #img = resize(img, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True)
    X_train[n] = img
print('Done!')

# Check if training data looks all right
ix = random.randint(0, len(y_train_ids))
imshow(np.squeeze(X_train[ix]))
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()



''' X_train.shape = (1200,520,696,1), Y_train.shape = (1200,520,696,1)'''




# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)






# Build U-Net model
inputs = Input((IMG_WIDTH, IMG_HEIGHT, 1))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy')#, metrics=[mean_iou])
#model.summary()





# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-ssc2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.3, batch_size=8, epochs=1, callbacks=[earlystopper, checkpointer])
results = model.fit(X_train, Y_train, validation_split=0.3, batch_size=8, epochs=1, callbacks=[earlystopper])




plt.plot(results.history['loss'][-10:])
plt.plot(results.history['val_loss'][-10:])
plt.title('Unet model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





os.chdir('/home/mingkuan/Dropbox/Unet/')
# Predict on train, val and test
model = load_model('model-ssc2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.7)], verbose=1)      
#X_train[:603].shape = (603,128,128,3), preds_train.shape = (603,128,128,1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.7):], verbose=1)

#preds_test = model.predict(X_test, verbose=1) #(2400, 256, 256, 1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8) # preds_train_t.shape =(603, 128, 128, 1) 
preds_val_t = (preds_val > 0.5).astype(np.uint8)
#preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
#preds_test_upsampled = []        # (65, 512, 680)
#for i in range(len(preds_test)):
#    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
#                                       (sizes_test[i][0], sizes_test[i][1]), 
#                                       mode='constant', preserve_range=True))

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(np.squeeze(X_train[ix]))
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()



# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(np.squeeze(X_train[int(X_train.shape[0]*0.7):][ix]))
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.7):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()






'''
Get masks for ssc data 
'''



'''Import test dataset(ssc 2019 case study data) for Unet to get masks before counting cells'''
from PIL import Image
import os
import numpy as np
import csv

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
    img = Image.open(i)
    ncols, nrows = img.size
    img = np.array(img.getdata()).reshape((nrows, ncols,1))
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True)
    traning_list.append(img)


X_test = traning_list
X_test = np.array(X_test)
X_test.shape

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
    img = Image.open(i)
    ncols, nrows = img.size
    img = np.array(img.getdata()).reshape((nrows, ncols,1))
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True)
    testing_list.append(img)
    
testing_list = np.array(testing_list)
testing_list.shape








os.chdir('/home/mingkuan/Dropbox/Unet/')
# Predict on train, val and test
model = load_model('model-ssc2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_test = model.predict(X_test, verbose=1) #(2400, 256, 256, 1)
preds_test_1 = model.predict(testing_list, verbose=1)

# Threshold predictions
x_train = (preds_test > 0.5).astype(np.uint8)
x_test = (preds_test_1 > 0.5).astype(np.uint8)
# Create list of upsampled test masks
#preds_test_upsampled = []        # (65, 512, 680)
#for i in range(len(preds_test)):
#    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
#                                       (sizes_test[i][0], sizes_test[i][1]), 
#                                       mode='constant', preserve_range=True))
    





np.save('train_x_resize.npy', X_test)   
np.save('train_x_mask.npy', x_train) 
np.save('test_x_resize.npy', testing_list)
np.save('test_x_mask.npy', x_test)

train_x_resize = np.load('train_x_resize.npy')
train_x_mask = np.load('train_x_mask.npy')
test_x_resize = np.load('test_x_resize.npy')
test_x_mask = np.load('test_x_mask.npy')


# erform a sanity check on some random test samples (ssc 2019 train data = 2400)
ix = random.randint(0, len(train_x_mask))
imshow(np.squeeze(train_x_resize[ix]))
plt.show()
imshow(np.squeeze(train_x_mask[ix]))
plt.show()
'''bad index=2242 '''

ix = random.randint(0, len(test_x_resize))
imshow(np.squeeze(test_x_resize[ix]))
plt.show()
imshow(np.squeeze(test_x_mask[ix]))
plt.show()






