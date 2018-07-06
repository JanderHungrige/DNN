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
def ResNet_deep_Beta(X_train,Y_train,Var):   

       def Block_unit(X_train,Var):
            def unit(x):
                 ident = x
                 x=layers.Bidirectional(LSTM(Var.hidden_units, activation=Var.activationF, return_sequences=True,   
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(x)
                 x=layers.Bidirectional(LSTM(Var.hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(x)                         
                 x=layers.Dropout(Var.dropout, noise_shape=(None, 1, Var.hidden_units*2))(x) 
                 x=layers.Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.))(x)                                                  
                 x = merge([ident,x], mode = 'sum') #mode 'sum' concat
                 return x
            return unit
                             
       def cake(Var):
              def unit(x):
                     for j in range(Var.residual_blocks):
                         x=Block_unit(X_train,Var)(x)              
                         return x                          
              return unit
    
       
       inp = Input(shape=(X_train.shape[1],X_train.shape[2]))
#       i = inp
       i=layers.Masking(mask_value=666,input_shape=(X_train.shape[1],X_train.shape[2]))(inp)
       i=layers.Dropout(Var.dropout/2, noise_shape=(None, 1, X_train.shape[2]))(i)
       i=layers.Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.))(i) 
       i=BatchNormalization(axis=1)(i)     
          
       i = cake(Var)(i) 
       i = cake(Var)(i) 
       i = cake(Var)(i)
       
               
       i=layers.Bidirectional(LSTM(Var.hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(i)                   
       i = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(i)

       model = Model(inputs=inp,outputs=i)  
       
       return model

#%%    
def ResNet_wide_Beta(X_train,Y_train,Var):   

       def Block_unit(X_train,Var,hidden_units):
            def unit(x):
                 ident = x
                 x=layers.Bidirectional(LSTM(hidden_units, activation=Var.activationF, return_sequences=True,   
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(x)
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(x)                                      
                 x=layers.Dropout(Var.dropout, noise_shape=(None, 1, hidden_units*2))(x) 
                 x=layers.Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.))(x)                                                  
                 x = merge([ident,x], mode = 'sum') #mode 'sum' concat
                 return x
            return unit
                             
       def cake(Var, hidden_units):
              def unit(x):
                     for j in range(Var.residual_blocks):
                         x=Block_unit(X_train,Var,hidden_units)(x)              
                         return x                          
              return unit
    

       inp = Input(shape=(X_train.shape[1],X_train.shape[2]))
#       i = inp
       i=layers.Masking(mask_value=666,input_shape=(X_train.shape[1],X_train.shape[2]))(inp)
       i=layers.Dropout(Var.dropout/2, noise_shape=(None, 1, X_train.shape[2]))(i)
       i=layers.Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.))(i) 
       intro_out=BatchNormalization(axis=1)(i)     
            
       Pfad1 = cake(Var,32)(intro_out) 
       Pfad1 = cake(Var,32)(Pfad1) 
       
       Pfad2 = cake(Var,2)(intro_out) 
       Pfad2 = cake(Var,2)(Pfad2) 

       Pfad3 = cake(Var,64)(intro_out) 
       Pfad3 = cake(Var,64)(Pfad3) 
       
       i = layers.concatenate([Pfad1, Pfad2, Pfad3])

       
               
       Outro_out=layers.Bidirectional(LSTM(Var.hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(i)                   
       Outro_out = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(Outro_out)

       model = Model(inputs=inp,outputs=Outro_out)  
       
       return model       

  
          