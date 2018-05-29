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


from keras.models import Sequential

from keras.layers import Dense, Activation
from keras.layers import LSTM, Activation
from keras.layers import Masking, Activation



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
def LSTM_model_1(X_train,Y_train,Dropout,hidden_units):
       
   model = Sequential()
#   model.add(Masking(mask_value=666, input_shape=X_train.shape))
#   model.add(LSTM(hidden_units, input_shape=(X_train.shape[1],X_train.shape[2]),activation='tanh', return_sequences=True, dropout=Dropout))   
   model.add(Masking(mask_value=666, input_shape=(X_train.shape[1],X_train.shape[2])))
   model.add(LSTM(hidden_units, activation='tanh', return_sequences=True, dropout=Dropout))   
   model.add(LSTM(hidden_units, return_sequences=True))
   model.add(LSTM(hidden_units, return_sequences=True))
   model.add(Dense(Y_train.shape[-1], activation='softmax'))

#   model.add(Activation('softmax'))

   model.compile(loss='mean_squared_error', optimizer='adam',metrics=['categorical_accuracy'])
     
   return model

#%%
def LSTM_model_2(X_train,Y_train,hidden_units,dropout):
   model = Sequential()
   model.add(Masking(mask_value=666, input_shape=X_train.shape))
   model.add(layers.Bidirectional(layers.LSTM(hidden_units, activation='tanh', return_sequences=True, dropout=dropout[0])))
   model.add(LSTM(hidden_units, return_sequences=True))
   model.add(LSTM(hidden_units, return_sequences=True))
   model.add(Dense(Y_test.shape[-1]), activation='softmax')

#   model.add(Activation('softmax'))

   model.compile(loss='mean_squared_error', optimizer='adam',metrics=['categorical_accuracy'])
     
   return model    
