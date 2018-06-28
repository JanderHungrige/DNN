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
def ResNet_LSTM_Beta(X_train,Y_train,dropout,hidden_units,activationF,residual_blocks):   

       def Block_unit(X_train,dropout,activationF,hidden_units):
            def unit(x):
                 ident = x
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)                                    
                 x=layers.Dropout(dropout, noise_shape=(None, 1, hidden_units*2))(x) 
                 x=layers.Dense(hidden_units, activation=activationF, kernel_constraint=max_norm(max_value=3.))(x)                                                  
                 x = merge([ident,x], mode = 'sum') #mode 'sum' concat
                 return x
            return unit
                             
       def cake(residual_blocks,hidden_units,X_train,dropout,activationF):
              def unit(x):
                     for j in range(residual_blocks):
                         x=Block_unit(X_train,dropout,activationF,hidden_units)(x)              
                         return x                          
              return unit
    
       
       inp = Input(shape=(X_train.shape[1],X_train.shape[2]))
#       i = inp
       i=layers.Masking(mask_value=666,input_shape=(X_train.shape[1],X_train.shape[2]))(inp)
       i=layers.Dropout(dropout/2, noise_shape=(None, 1, X_train.shape[2]))(i)
       i=layers.Dense(34, activation=activationF, kernel_constraint=max_norm(max_value=3.))(i) 
       i=BatchNormalization(axis=1)(i)     
          
       i = cake(residual_blocks,2,X_train,dropout,activationF)(i) 
       i = cake(residual_blocks,32,X_train,dropout,activationF)(i) 
       i = cake(residual_blocks,64,X_train,dropout,activationF)(i)
       
               
       i=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(i)                   
       i = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(i)

       model = Model(inputs=inp,outputs=i)  
       
       return model
#%%   
def ResNet_LSTM_1(X_train,Y_train,dropout,hidden_units,activationF,residual_blocks):   

       def Block_unit(which,X_train,dropout,activationF,hidden_units,x):
            def intro(x,X_train,activationF):
                 x=layers.Masking(mask_value=666,input_shape=(X_train.shape[1],X_train.shape[2]))(x)
                 x=layers.Dropout(dropout/2, noise_shape=(None, 1, X_train.shape[2]))(x)
                 x=layers.Dense(34, activation=activationF, kernel_constraint=max_norm(max_value=3.))(x) 
                 x=BatchNormalization(axis=1)(x)         
                 return x
            
            def residual_LSTM_block(X_train,hidden_units,dropout):
                 ident = x
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)                   
                 x=layers.Dropout(dropout, noise_shape=(None, 1, hidden_units*2))(x) 
                 x=layers.Dense(hidden_units, activation=activationF, kernel_constraint=max_norm(max_value=3.))(x)                                  
#                 x = merge([ident,x],mode='sum')
                 return x

            def outro(X_train,Y_train,hidden_units,dropout):
                 ident = x
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)                   
                 x=layers.Dropout(dropout, noise_shape=(None, 1, hidden_units*2))(x) 
                 x=layers.Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(x)
#                 x = merge([ident,x],mode='sum')
            return x         
                             
            if which==0:
                 out=intro(x,X_train,activationF)(x)
            if which == 1337:
                 out=outro(x,X_train,Y_train,hidden_units,dropout)(x)
            else:
                out=residual_LSTM_block(x,X_train,hidden_units,dropout)(x)                  

            return out
            
       def cake(residual_blocks,hidden_units,X_train,dropout,activationF):
              def unit(x):
                     for j in range(residual_blocks):
                            if residual_blocks==1:
                                   x=Block_unit(j,X_train,dropout,activationF,hidden_units,x)(x)  
                                   x=Block_unit(1,X_train,dropout,activationF,hidden_units,x)(x)                            
                                   x=Block_unit(1337,X_train,dropout,activationF,hidden_units,x)(x)
                            else:
                                   if i!=(residual_blocks-1): 
                                          x=Unit(j,X_train,dropout,activationF,x,hidden_units)(x)  
                                   else:
                                          x=Unit(1337,X_train,dropout,activationF,x,hidden_units)(x)
                            return x
                           
              return unit
    
       
       inp = Input(shape=(X_train.shape[1],X_train.shape[2]))
#       i = inp
              
       i = cake(residual_blocks,2,X_train,dropout,activationF)(inp) 
       i = cake(residual_blocks,32,X_train,dropout,activationF)(i) 
       i = cake(residual_blocks,64,X_train,dropout,activationF)(i)
       
       i=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(i)                   
       i = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(i)

       model = Model(inputs=inp,outputs=i)  
       
       return model
       
#%%       
       
def ResNeXt_LSTM_1(x):
       def Add_common_layers(y,activationF):
              y=layers.Masking(mask_value=666,input_shape=(y.shape[1],y.shape[2]))(y)
              y=layers.Dropout(0.2, noise_shape=(None, 1, y.shape[2]))(y)
              y=layers.Dense(32, activation=activationF, kernel_constraint=max_norm(max_value=3.))(y)
              return y
       
       def grouped_LSTM(y,hidden_units,dropout,NR_Wege):
              if cardinality == 1:
                     y=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(y)
                     y=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(y)
                     return y
                     assert not NR_Wege % cardinality
              _d = NR_Wege // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
              groups = []
              for j in range(cardinality):
                  group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
                  groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
              y = layers.concatenate(groups)
              return y
              
       def end(y):
              y=layers.Dropout()
              y=layers.Dense()

   
  
          