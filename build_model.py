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



def basic_dense_model(X_train,Y_train):   
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu',input_shape=(X_train.shape[1],)))             
    model.add(layers.Dense(16, activation='relu'))           
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',#'Adadelta',
                  metrics=['categorical_accuracy'])#['accuracy'])


    return model
#%%
def LSTM_model_1(X_train,Y_train,X_val,Y_val,hidden_units,use_dropout,dropout):
   model = Sequential()
   model.add(LSTM(hidden_units, activation='tanh', input_shape=(X_train.shape)))
   model.add(LSTM(hidden_units, return_sequences=True))
   model.add(LSTM(hidden_units, return_sequences=True))
   if use_dropout:
       model.add(Dropout(dropout))
   model.add(Activation('softmax'))

   model.compile(loss='mean_square_error', optimizer='adam',metrics=['categorical_accuracy'])
     
   return model

#%%       
