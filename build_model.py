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
from keras import regularizers

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Masking
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Bidirectional
from keras.layers import BatchNormalization

from keras.constraints import max_norm


#%%
def basic_dense_model(X_train,Y_train):   
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu',input_shape=(X_train.shape[1],)))             
    model.add(layers.Dense(16, activation='relu'))           
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))

#    model.compile(loss='categorical_crossentropy',
#                  optimizer='rmsprop',#'Adadelta',
#                  metrics=['categorical_accuracy'])#['accuracy'])

    return model
#%%
def LSTM_model_1_gen(X_train,Y_train,Var):
       
   model = Sequential()
   model.add(Masking(mask_value=666, input_shape=(X_train.shape[1],X_train.shape[2])))
   model.add(LSTM(Var.hidden_units, activation=Var.activationF, return_sequences=True, dropout=Var.dropout))   
   model.add(LSTM(Var.hidden_units, return_sequences=True))
   model.add(LSTM(Var.hidden_units, return_sequences=True))
   model.add(Dense(Y_train.shape[-1], activation='softmax'))

   return model

#%%
def LSTM_model_1(X_train,Y_train,Var):
       
   model = Sequential()
#   model.add(Masking(mask_value=666, input_shape=X_train.shape))
#   model.add(LSTM(hidden_units, input_shape=(X_train.shape[1],X_train.shape[2]),activation='tanh', return_sequences=True, dropout=Dropout))   
   model.add(Masking(mask_value=666, input_shape=(X_train.shape[1],X_train.shape[2])))
   model.add(LSTM(Var.hidden_units, activation=Var.activationF, return_sequences=True, dropout=Var.dropout))  
#   model.add(LSTM(hidden_units, activation='sigmoid', return_sequences=True, dropout=dropout))      
   model.add(LSTM(Var.hidden_units, return_sequences=True))
   model.add(LSTM(Var.hidden_units, return_sequences=True))
   model.add(Dense(Y_train.shape[-1], activation='softmax'))

   return model
#%%
def LSTM_model_2(X_train,Y_train,Var):       
   model = Sequential()
   model.add(Masking(mask_value=666, input_shape=(X_train.shape[1],X_train.shape[2]) ))
   model.add(Dropout(Var.dropout, noise_shape=(None, 1, X_train.shape[2]) ))   
   model.add(Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=2.) ))   #dense unit was 34
   model.add(LSTM(Var.hidden_units, return_sequences=True, dropout=Var.dropout, recurrent_dropout=Var.dropout))  
   model.add(LSTM(Var.hidden_units, return_sequences=True, dropout=Var.dropout, recurrent_dropout=Var.dropout))
#   model.add(LSTM(hidden_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))
   model.add(Dense(Y_train.shape[-1], activation='softmax'))

   return model


#%%
def model_3_LSTM(X_train,Y_train,Var):
   model = Sequential()
   model.add(Masking(mask_value=666, input_shape=(X_train.shape[1],X_train.shape[2]) ))
   model.add(Dropout(Var.dropout, noise_shape=(None, 1, X_train.shape[2]) ))   
   model.add(layers.Bidirectional(layers.LSTM(Var.hidden_units, activation=Var.activationF, return_sequences=True, dropout=Var.dropout)), merge_mode='concat')
   model.add(layers.Bidirectional(layers.LSTM(Var.hidden_units, return_sequences=True, dropout=Var.dropout)), merge_mode='concat')

#   model.add(LSTM(hidden_units, return_sequences=True))
#   model.add(LSTM(hidden_units, return_sequences=True))
   model.add(Dense(Y_test.shape[-1]), activation='softmax')

#   model.add(Activation('softmax'))

   return model  
#%%
def model_3_LSTM_advanced(X_train,Y_train,Var):   
   maxnorm=3.
   batch_size=X_train.shape[0]
   n_frames=X_train.shape[2]
   model = Sequential()
   model.add(Masking(mask_value=666, input_shape=(X_train.shape[1],X_train.shape[2])))
   model.add(Dropout(0.2, noise_shape=(None, 1, X_train.shape[2]) ))   
   model.add(Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.)))
   model.add(Bidirectional(LSTM(Var.hidden_units, return_sequences=True,   
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout)))
   model.add(Bidirectional(LSTM(Var.hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout)))   
   model.add(Dropout(0.5, noise_shape=(None, 1, Var.hidden_units*2)))
   model.add(Dense(Y_train.shape[-1], activation='softmax', kernel_constraint=max_norm(max_value=3.)))
   model.summary()
   
   return model  

#%%
def model_3_LSTM_advanced_no_bi(X_train,Y_train,Var):   
   maxnorm=3.
   batch_size=X_train.shape[0]
   n_frames=X_train.shape[2]
   model = Sequential()
   model.add(Masking(mask_value=666, input_shape=(X_train.shape[1],X_train.shape[2])))
   model.add(Dropout(0.2, noise_shape=(None, 1, X_train.shape[2]) ))   
   model.add(Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.)))
   model.add(LSTM(Var.hidden_units, return_sequences=True,   
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout))
   model.add(LSTM(Var.hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout))
   model.add(Dropout(0.5, noise_shape=(None, 1, Var.hidden_units*2)))
   model.add(Dense(Y_train.shape[-1], activation='softmax', kernel_constraint=max_norm(max_value=3.)))
   model.summary()
   
   return model     
#%%
def model_4_GRU(X_train,Y_train,Var):   
   model = Sequential()
   model.add(Masking(mask_value=666, input_shape=(X_train.shape[1],X_train.shape[2])))
   model.add(Dropout(0.2, noise_shape=(None, 1, X_train.shape[2]) ))   
   model.add(Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.)))
   model.add(Bidirectional(GRU(Var.hidden_units, return_sequences=True,   
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.),
                                dropout=Var.dropout, recurrent_dropout=Var.dropout)))
   model.add(Bidirectional(GRU(Var.hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout)))
   model.add(Dropout(0.5, noise_shape=(None, 1, Var.hidden_units*2)))
   model.add(Dense(Y_train.shape[-1], activation='softmax', kernel_constraint=max_norm(max_value=3.)))
   model.summary()
   
   return model   

#%%
def model_4_GRU_advanced(X_train,Y_train,Var):   
   model = Sequential()
   model.add(Masking(mask_value=666, input_shape=(X_train.shape[1],X_train.shape[2])))
   model.add(Dropout(0.2, noise_shape=(None, 1, X_train.shape[2]) ))   
   model.add(Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.)))
   model.add(GRU(Var.hidden_units, return_sequences=True,   
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))
   model.add(GRU(Var.hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))
   model.add(Dropout(0.5, noise_shape=(None, 1, Var.hidden_units*2)))
   model.add(Dense(Y_train.shape[-1], activation='softmax', kernel_constraint=max_norm(max_value=3.)))
   model.summary()
   
   return model   





   
#%% ORIGINAL FROM arnaud.moreau@philips.com
   #---------------------------------------------------------------------------------------
   #---------------------------------------------------------------------------------------
   
   #---------------------------------------------------------------------------------------
   #---------------------------------------------------------------------------------------
def LSTM_model_3_original(X_train,Y_train,dropout,hidden_units,MaskWert):   
   model = Sequential()
   model.add(Masking(input_shape=(max_length,)))
   model.add(Dropout(0.2, noise_shape=(batch_size, 1, n_frames)))
   model.add(Dense(32, activation='sigmoid', kernel_constraint=maxnorm(max_norm)))
   model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_constraint=maxnorm(max_value=3.), dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
   model.add(Dropout(0.5, noise_shape=(batch_size, 1, 128)))
   model.add(Dense(n_classes, activation='softmax', kernel_constraint=maxnorm(max_norm)))
   model.summary()
   
#   model.compile(loss='mean_squared_error', optimizer='adam',metrics=['categorical_accuracy'],sample_weight_mode="temporal")
   return model