#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 23:06:16 2018

@author: 310122653
"""

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
from keras.models import Model

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Masking
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers import Input

from keras.constraints import max_norm


#%%
def model_3_LSTM_advanced1(X_train,Y_train,Var):   
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
def model_4_GRU_advanced1(X_train,Y_train,Var):   
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
def model_3_LSTM_advanced2(X_train,Y_train,Var):   
   maxnorm=3.
   batch_size=X_train.shape[0]
   n_frames=X_train.shape[2]
   model = Sequential()
   model.add(Masking(mask_value=666, input_shape=(X_train.shape[1],X_train.shape[2])))
   model.add(Dropout(0.2, noise_shape=(None, 1, X_train.shape[2]) ))   
   model.add(Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.)))
   model.add(Bidirectional(LSTM(Var.hidden_units, return_sequences=True,   
                                kernel_regularizer=regularizers.l1(0.1),
                                activity_regularizer=regularizers.l1(0.1),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout)))
   model.add(Bidirectional(LSTM(Var.hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l1(0.1),
                                activity_regularizer=regularizers.l1(0.1),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout)))  
   model.add(Bidirectional(LSTM(Var.hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l1(0.1),
                                activity_regularizer=regularizers.l1(0.1),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout))) 
   model.add(Bidirectional(LSTM(Var.hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l1(0.1),
                                activity_regularizer=regularizers.l1(0.1),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout)))    
   model.add(Dropout(0.5, noise_shape=(None, 1, Var.hidden_units*2)))
   model.add(Dense(Y_train.shape[-1], activation='softmax', kernel_constraint=max_norm(max_value=3.)))
   model.summary()
   
   return model  

#%%
def model_4_GRU_advanced2(X_train,Y_train,Var):   
   model = Sequential()
   model.add(Masking(mask_value=666, input_shape=(X_train.shape[1],X_train.shape[2])))
   model.add(Dropout(0.2, noise_shape=(None, 1, X_train.shape[2]) ))   
   model.add(Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.)))
   model.add(Bidirectional(GRU(Var.hidden_units, return_sequences=True,   
                                kernel_regularizer=regularizers.l1(0.3),
                                activity_regularizer=regularizers.l1(0.3),
                                kernel_constraint=max_norm(max_value=3.),
                                dropout=Var.dropout, recurrent_dropout=Var.dropout)))
   model.add(Bidirectional(GRU(Var.hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l1(0.3),
                                activity_regularizer=regularizers.l1(0.3),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout)))
   model.add(Bidirectional(GRU(Var.hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l1(0.3),
                                activity_regularizer=regularizers.l1(0.3),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout)))   
   model.add(Bidirectional(GRU(Var.hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l1(0.3),
                                activity_regularizer=regularizers.l1(0.3),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout)))     
   model.add(Dropout(0.5, noise_shape=(None, 1, Var.hidden_units*2)))
   model.add(Dense(Y_train.shape[-1], activation='softmax', kernel_constraint=max_norm(max_value=3.)))
   model.summary()
   
   return model   