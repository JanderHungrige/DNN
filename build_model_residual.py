# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:46:27 2018

@author: 310122653
"""

import keras 
#import keras.backend as K
#import tensorflow as tf

from keras.models import Model

from keras import optimizers 
from keras import losses
from keras import metrics
from keras import models
from keras import layers
from keras import callbacks
from keras import regularizers

from keras.utils import np_utils

from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import merge
#from keras.layers import Lambda
from keras.layers import Bidirectional
from keras.layers import BatchNormalization

from keras.constraints import max_norm
#%%   
def ResNet_deep_Beta(X_train,Y_train,dropout,hidden_units,Dense_Unit,activationF,residual_blocks,Kr,Ar):   

       def Block_unit(X_train,dropout,activationF,hidden_units,Dense_Unit,Kr,Ar):
            def unit(x):
                 ident = x
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True,   
                                kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l2(0.01),
                                kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l2(0.01),
                                kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)                         
                 x=layers.Dropout(dropout, noise_shape=(None, 1, hidden_units*2))(x) 
                 x=layers.Dense(Dense_Unit, activation=activationF, kernel_constraint=max_norm(max_value=3.))(x)                                                  
                 x = merge([ident,x], mode = 'sum') #mode 'sum' concat
                 return x
            return unit
                             
       def cake(residual_blocks,hidden_units,X_train,dropout,activationF):
              def unit(x):
                     for j in range(residual_blocks):
                         x=Block_unit(X_train,dropout,activationF,hidden_units,Dense_Unit,Kr,Ar)(x)              
                         return x                          
              return unit
    
       
       inp = Input(shape=(X_train.shape[1],X_train.shape[2]))
#       i = inp
       i=layers.Masking(mask_value=666,input_shape=(X_train.shape[1],X_train.shape[2]))(inp)
       i=layers.Dropout(dropout/2, noise_shape=(None, 1, X_train.shape[2]))(i)
       i=layers.Dense(Dense_Unit, activation=activationF, kernel_constraint=max_norm(max_value=3.))(i) 
       i=BatchNormalization(axis=1)(i)     
          
       i = cake(residual_blocks,2,X_train,dropout,activationF)(i) 
       i = cake(residual_blocks,32,X_train,dropout,activationF)(i) 
       i = cake(residual_blocks,64,X_train,dropout,activationF)(i)
       
               
       i=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(i)                   
       i = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(i)

       model = Model(inputs=inp,outputs=i)  
       
       return model

#%%    
def ResNet_wide_Beta(X_train,Y_train,dropout,hidden_units,Dense_Unit,activationF,residual_blocks,Kr,Ar):   

       def Block_unit(X_train,dropout,activationF,hidden_units,Dense_Unit,Kr,Ar):
            def unit(x):
                 ident = x
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True,   
                                kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l2(0.01),
                                kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)
                 x=lyers.Bidirectional(LSTM(hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l2(0.01),
                                kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)                                      
                 x=layers.Dropout(dropout, noise_shape=(None, 1, hidden_units*2))(x) 
                 x=layers.Dense(Dense_Unit, activation=activationF, kernel_constraint=max_norm(max_value=3.))(x)                                                  
                 x = merge([ident,x], mode = 'sum') #mode 'sum' concat
                 return x
            return unit
                             
       def cake(residual_blocks,hidden_units,X_train,dropout,activationF):
              def unit(x):
                     for j in range(residual_blocks):
                         x=Block_unit(X_train,dropout,activationF,hidden_units,Dense_Unit,Kr,Ar)(x)              
                         return x                          
              return unit
    

       inp = Input(shape=(X_train.shape[1],X_train.shape[2]))
#       i = inp
       i=layers.Masking(mask_value=666,input_shape=(X_train.shape[1],X_train.shape[2]))(inp)
       i=layers.Dropout(dropout/2, noise_shape=(None, 1, X_train.shape[2]))(i)
       i=layers.Dense(Dense_Unit, activation=activationF, kernel_constraint=max_norm(max_value=3.))(i) 
       intro_out=BatchNormalization(axis=1)(i)     
            
       Pfad1 = cake(residual_blocks,32,X_train,dropout,activationF)(intro_out) 
       Pfad1 = cake(residual_blocks,32,X_train,dropout,activationF)(Pfad1) 
       
       Pfad2 = cake(residual_blocks,2,X_train,dropout,activationF)(intro_out) 
       Pfad2 = cake(residual_blocks,2,X_train,dropout,activationF)(Pfad2) 

       Pfad3 = cake(residual_blocks,64,X_train,dropout,activationF)(intro_out) 
       Pfad3 = cake(residual_blocks,64,X_train,dropout,activationF)(Pfad3) 
       
       i = layers.concatenate([Pfad1, Pfad2, Pfad3])

       
               
       Outro_out=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(i)                   
       Outro_out = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(Outro_out)

       model = Model(inputs=inp,outputs=Outro_out)  
       
       return model       

  
          