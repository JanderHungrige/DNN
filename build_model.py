# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:16:22 2018

@author: 310122653
"""       

import keras
from keras import optimizers 
from keras import losses
from keras import metrics
from keras import models
from keras import layers
from keras import callbacks
from keras.utils import np_utils



def basic_dense_model(X_train,y_train):   
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu',input_shape=(X_train.shape[1],)))             
    model.add(layers.Dense(16, activation='relu'))           
    model.add(layers.Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',#'Adadelta',
                  metrics=['categorical_accuracy'])#['accuracy'])


    return model
#%%
def RNN_model(X_train,y_train):
    model = models.Sequential()
    
    model=build_model()

        
    return model

#%%       
