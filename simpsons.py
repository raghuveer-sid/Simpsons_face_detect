# this model does not have good accuracy 

from cProfile import label
import importlib
import os
import caer
import canaro
from more_itertools import callback_iter
import numpy as np
import cv2 as cv
import gc
import tensorflow


# Changing all images to same size before feed them to code for this simpsons dataset 80,80 is good
img_size = (80,80)
# no.of channels  since we dont deed color w set it to 1 i.e grayscale
channels = 1
# path

char_path = r'D:/projects/opencv_proj/datasets/simpsons_dataset'

#check number of images in each folder by making a dictionary and sorting them by numbers

char_dict = {}
# STORING IN DICTIONARY
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path,char)))

# SORTING

char_dict = caer.sort_dict(char_dict,descending=True)
# print(char_dict)

characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break

print(characters)

# Training data using caer

train = caer.preprocess_from_dir(char_path,characters,channels=channels,IMG_SIZE=img_size,isShuffle=True)

# now train is just a list
# seperate train to lables and features

featureSet,labels = caer.sep_train(train,IMG_SIZE=img_size)

# normalize the feature set it network will lear faster by this process

featureSet = caer.normalize(featureSet)

# we do not need to normalize lables but we need to convert them from num integers to binary class vectors

from tensorflow.python.keras.utils.np_utils import to_categorical


labels = to_categorical(labels,len(characters))


# train validation data using caer

x_train,x_val,y_train,y_val = caer.train_val_split(featureSet,labels,val_ratio=0.2)

# to save memory we delete variable we are not using
del train
del featureSet
del labels
gc.collect()

# image generator

batch_size = 32
epochs = 10

datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train,batch_size=batch_size)

# creating a model 
model = canaro.models.createSimpsonsModel(IMG_SIZE=img_size,channels= channels,output_dim=len(characters),loss='binary_crossentropy',
                                          decay=1e-6,learning_rate=0.001,momentum=0.9,nesterov=True)

model.summary()

# call back list conataining learning rate to learn better
from tensorflow.python.keras.callbacks import LearningRateScheduler

callback_list = [LearningRateScheduler(canaro.lr_schedule)]

# trai model

train = model.fit(train_gen,steps_per_epoch = len(x_train)//batch_size,epochs=epochs,validation_data = (x_val,y_val),validation_steps = len(y_val)//batch_size,callbacks=callback_list)


# testing stage

test_path = r'D:/projects/opencv_proj/datasets/kaggle_simpson_testset/kaggle_simpson_testset/bart_simpson_13.jpg'
img = cv.imread(test_path)
cv.imshow('img',img)

# prepare images to fit model 

def prepare(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.resize(img,img_size)
    img = caer.reshape(img,img_size,1)
    return img

predictions = model.predict(prepare(img))

print(characters[np.argmax(predictions[0])])

