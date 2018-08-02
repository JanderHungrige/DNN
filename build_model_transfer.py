# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:23:54 2018

@author: 310122653
"""

import keras 
#import keras.backend as K
#import tensorflow as tf

from keras.models import Model

#from keras import optimizers 
#from keras import losses
#from keras import metrics
#from keras import models
from keras import layers
#from keras import callbacks
from keras import regularizers

#from keras.utils import np_utils

from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU

from keras.layers import Masking
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import merge
#from keras.layers import Lambda
from keras.layers import Bidirectional
from keras.layers import BatchNormalization

from keras.constraints import max_norm
from keras.models import model_from_json


def Transfer_wide_Beta_GRU(X_train,Y_train,Var):
      
       def ModelLaden(destination,modelname,modelweights):
              json_file = open(destination+modelname+'.json', 'r')
              loaded_model_json = json_file.read()
              json_file.close()
              loaded_model = model_from_json(loaded_model_json)
              # load weights into new model
              loaded_model.load_weights(destination+modelweights+'.h5')
              return loaded_model       

       def Block_unit(X_train,Var,hidden_units):
            def unit(x):
                 ident = x
                 x=layers.Bidirectional(GRU(hidden_units, activation=Var.activationF, return_sequences=True,   
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(x)
                 x=layers.Bidirectional(GRU(hidden_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(x)                                      
                 x=layers.Dropout(Var.dropout, noise_shape=(None, 1, hidden_units*2))(x) 
                 x=layers.Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.))(x)                                                  
                 x=layers.add([ident,x]) 
                 return x
            return unit
                             
       def cake(Var, hidden_units):
              def unit(x):
                     for j in range(Var.residual_blocks):
                         x=Block_unit(X_train,Var,hidden_units)(x)              
                         return x                          
              return unit
       
#       def extraingredient(Var,destiantion,modelname,modelweights):
#              loaded_model=ModelLaden(destiantion,modelname,modelweights)
#              def unit(x,loaded_model):
#                     ident = x
##                     x=loaded_model(x)
#                     return x
#              return unit
       destination=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/')
       modelname='18_AS_QS_MMCcECG_ResWGRUmodel'
       modelweights='18_AS_QS_MMCcECG_ResWGRUmodel_weigths'               
#    
       Bimodel=ModelLaden(destination,modelname,modelweights)
       
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
       

       
#       Pfad4 = extraingredient(Var,destiantion,modelname,modelweights)(intro_out) 
       
       i = layers.concatenate([Pfad1, Pfad2, Pfad3])

#       Outro_out=layers.Bidirectional(GRU(Var.hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(i)                   
#       Outro_out = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(Outro_out)
       Outro_out = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(i)

       model = Model(inputs=inp,outputs=Outro_out)  
       
       return model       