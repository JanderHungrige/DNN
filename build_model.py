# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:16:22 2018

@author: 310122653
"""       

import keras 
import keras.backend as K
import tensorflow as tf

from keras import optimizers 
from keras import losses
from keras import metrics
from keras import models
from keras import layers
from keras import callbacks
from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Lambda

from keras.constraints import max_norm



#%%
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
def LSTM_model_1_gen(lookback,Nr_Features,Nr_labels,dropout,hidden_units,MaskWert):
       
   model = Sequential()
   model.add(Masking(mask_value=MaskWert, input_shape=(lookback,Nr_Features)))
   model.add(LSTM(hidden_units, activation='tanh', return_sequences=True, dropout=dropout))   
   model.add(LSTM(hidden_units, return_sequences=True))
   model.add(LSTM(hidden_units, return_sequences=True))
   model.add(Dense(Nr_labels, activation='softmax'))

   model.compile(loss='mean_squared_error', optimizer='adam',metrics=['categorical_accuracy'],sample_weight_mode="temporal")    
   return model

#%%
def LSTM_model_1(X_train,Y_train,dropout,hidden_units,MaskWert):
       
   model = Sequential()
#   model.add(Masking(mask_value=666, input_shape=X_train.shape))
#   model.add(LSTM(hidden_units, input_shape=(X_train.shape[1],X_train.shape[2]),activation='tanh', return_sequences=True, dropout=Dropout))   
   model.add(Masking(mask_value=MaskWert, input_shape=(X_train.shape[1],X_train.shape[2])))
   model.add(LSTM(hidden_units, activation='tanh', return_sequences=True, dropout=dropout))  
#   model.add(LSTM(hidden_units, activation='sigmoid', return_sequences=True, dropout=dropout))      
   model.add(LSTM(hidden_units, return_sequences=True))
   model.add(LSTM(hidden_units, return_sequences=True))
   model.add(Dense(Y_train.shape[-1], activation='softmax'))

   model.compile(loss='mean_squared_error', optimizer='adam',metrics=['categorical_accuracy'],sample_weight_mode="temporal")    
   return model
#%%
def LSTM_model_2(X_train,Y_train,dropout,hidden_units,MaskWert):       
   model = Sequential()
   model.add(Masking(mask_value=MaskWert, input_shape=(X_train.shape[1],X_train.shape[2]) ))
   model.add(Dropout(dropout, noise_shape=(None, 1, X_train.shape[2]) ))   
   model.add(Dense(32, activation='tanh', kernel_constraint=max_norm(max_value=4.) ))   
   model.add(LSTM(hidden_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))  
   model.add(LSTM(hidden_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))
#   model.add(LSTM(hidden_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))
   model.add(Dense(Y_train.shape[-1], activation='softmax'))

   model.compile(loss='mean_squared_error', optimizer='adam',metrics=['categorical_accuracy'],sample_weight_mode="temporal")    
   return model


#%%
def LSTM_model_3(X_train,Y_train,dropout,hidden_units,MaskWert):
   model = Sequential()
#   model.add(Masking(mask_value=MaskWert, input_shape=X_train.shape))
   model.add(layers.Bidirectional(layers.LSTM(hidden_units, activation='tanh', return_sequences=True, dropout=dropout)))
   model.add(LSTM(hidden_units, return_sequences=True))
   model.add(LSTM(hidden_units, return_sequences=True))
   model.add(Dense(Y_test.shape[-1]), activation='softmax')

#   model.add(Activation('softmax'))

   model.compile(loss='mean_squared_error', optimizer='adam',metrics=['categorical_accuracy'],sample_weight_mode="temporal")     
   return model  
#%%
def LSTM_model_3_advanced(X_train,Y_train,dropout,hidden_units,MaskWert):   
   batch_size=X_train.shape[0]
   n_frames=X_train.shape[2]
   model = Sequential()
   model.add(Masking(mask_value=MaskWert, input_shape=(X_train.shape[1],X_train.shape[2])))
   model.add(Dropout(0.2, noise_shape=(batch_size, 1, n_frames)))
   model.add(Dense(32, activation='sigmoid', kernel_constraint=maxnorm(max_norm)))
   model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_constraint=maxnorm(max_norm), dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
   model.add(Dropout(0.5, noise_shape=(batch_size, 1, 128)))
   model.add(Dense(Y_train.shape[-1], activation='softmax', kernel_constraint=maxnorm(max_norm)))
   model.summary()
   
   model.compile(loss='mean_squared_error', optimizer='adam',metrics=['categorical_accuracy'],sample_weight_mode="temporal")
   return model  
   
#%% ORIGINAL FROM arnaud.moreau@philips.com
def LSTM_model_3_original(X_train,Y_train,dropout,hidden_units,MaskWert):   
   model = Sequential()
   model.add(Masking(input_shape=(max_length,)))
   model.add(Dropout(0.2, noise_shape=(batch_size, 1, n_frames)))
   model.add(Dense(32, activation='sigmoid', kernel_constraint=maxnorm(max_norm)))
   model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_constraint=maxnorm(max_norm), dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
   model.add(Dropout(0.5, noise_shape=(batch_size, 1, 128)))
   model.add(Dense(n_classes, activation='softmax', kernel_constraint=maxnorm(max_norm)))
   model.summary()
   
   model.compile(loss='mean_squared_error', optimizer='adam',metrics=['categorical_accuracy'],sample_weight_mode="temporal")
   return model