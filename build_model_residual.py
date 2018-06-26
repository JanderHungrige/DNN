# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:46:27 2018

@author: 310122653
"""

import keras 
import keras.backend as K
import tensorflow as tf

from keras.models import Model

from keras import Input 
from keras import optimizers 
from keras import losses
from keras import metrics
from keras import models
from keras import layers
from keras import callbacks
from keras.utils import np_utils

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import Dropout
from keras.layers import Activation
#from keras.layers import Lambda
from keras.layers import Bidirectional
from keras.layers import BatchNormalization

from keras.constraints import max_norm

   maxnorm=3.
   batch_size=X_train.shape[0]
   n_frames=X_train.shape[2]

   
def ResNet_LSTM_1(X_train,Y_train,dropout,hidden_units,activationF):   

       def Unit(which,X_train,dropout,activationF,x,hidden_units,dropout):
            def intro(X_train,activationF):
                 x=layers.Masking(mask_value=666,input_shape=X_train.shape[1],X_train.shape[2])(x)
                 x=layers.Dropout(dropout/2, noise_shape=(None, 1, X_train.shape[2]))(x)
                 x=layers.Dense(32, activation=activationF, kernel_constraint=max_norm(max_value=3.))(x) 
                 x=BatchNormalization(axis=1)(x)         
                 return x
            
            def residual_LSTM_block(x,hidden_units,dropout):
                 ident = x
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)                   
                 x=layers.Dropout(dropout, noise_shape=(None, 1, 64))(x) 
                 x=layers.Dense(32, activation=activationF, kernel_constraint=max_norm(max_value=3.))(x)                                  
#                 x = merge([ident,x],mode='sum')
                 return x

            def outro(x,Y_train,hidden_units,dropout):
                 ident = x
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)
                 x=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(x)                   
                 x=layers.Dropout(dropout, noise_shape=(None, 1, 64))(x) 
                 x=layers.Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(x)
#                 x = merge([ident,x],mode='sum')
                 return x         
                             
            if which==0:
                 out=intro(X_train,activationF)
            if which == 1337:
                 out=outro(x,Y_train,hidden_units,dropout)
            else:
                out=residual_LSTM_block(x,hidden_units,dropout)                     

            return out
            
       def cake(residual_blocks,hidden_units,X_train,dropout,activationF):
            for i in range(residual_blocks):
                   if i != len(range(residual_blocks)-1) : 
                               x=Unit(i,X_train,dropout,activationF,x,hidden_units,dropout)(x)  
                   else:
                               x=Unit(1337,X_train,dropout,activationF,x,hidden_units,dropout)(x)
                           
            return x
    
       
       inp = Input(shape=(X_train.shape[1],X_train.shape[2]))
       i = inp
              
       i = cake(residual_blocks,2,X_train,dropout,activationF)(i) 
       i = cake(residual_blocks,32,X_train,dropout,activationF)(i) 
       i = cake(residual_blocks,64,X_train,dropout,activationF)(i)
       
       i=layers.Bidirectional(LSTM(hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=dropout, recurrent_dropout=dropout))(i)                   
       i = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(i)

       model = Model(input=inp,output=i)  
       
       return model
       
#%%       
       
def ResNeXt_LSTM_1(x):
       def Add_common_layers(y,activationF):
              y=layers.Masking(mask_value=666,input_shape=y.shape[1],y.shape[2])(y)
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

   
   model.add(Masking(mask_value=666, input_shape=(X_train.shape[1],X_train.shape[2])))
   model.add(Dropout(0.2, noise_shape=(None, 1, X_train.shape[2]) ))   
          