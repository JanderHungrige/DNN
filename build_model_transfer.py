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


#%%      
def Transfer_wide_Beta_GRU(X_train,Y_train,Var):
# Here the whole model is loaded, fixed and then built into the residual as a path
       # Problem. submodels as layers have a problem with masking : 
#       https://github.com/keras-team/keras/issues/3524
#       https://github.com/keras-team/keras/issues/6541
       # PRobable solution is to do it with get_weights; set_weights, See next function
       
       
       class ASQSModel:
              destination=('C:/Users/310122653/Documents/GitHub/DNN/Results/')
              modelname='4_TEst_to_get_weightsmodel'
              modelweights='4_TEst_to_get_weightsmodel_weigths'  
       
       class ASISModel:
              destination=('C:/Users/310122653/Documents/GitHub/DNN/Results/')
              modelname='4_TEst_to_get_weightsmodel'
              modelweights='4_TEst_to_get_weightsmodel_weigths'     
       
       class ASCTWModel:
              destination=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/')
              modelname=''
              modelweights=''  
       
       class QSISModel:
              destination=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/')
              modelname=''
              modelweights=''          
       
       class QSCTWModel:
              destination=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/')
              modelname=''
              modelweights=''
       
       class ISCTWModel:
              destination=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/')
              modelname=''
              modelweights=''          

           
       def ModelLaden(destination,modelname):
              json_file = open(destination+modelname+'.json', 'r')
              loaded_model_json = json_file.read()
              json_file.close()
              loaded_model = model_from_json(loaded_model_json)
              # load weights into new model
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
       
       
#       def extraingredient(Var,loaded_model):
#              def BiUnit(x,loaded_model):
#                     ident = x
#                     x=loaded_model(x)
#                     return x                 
#              return BiUnit
              

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
       
       loaded_model=ModelLaden(ASQSModel.destination,ASQSModel.modelname)
       loaded_model.load_weights(ASQSModel.destination+ASQSModel.modelweights+'.h5') 
#       loaded_model.trainable = False  
#       loaded_model.support_masking=True 
#       for layer in loaded_model.layers[1:6]:  #0-6 is all 0:4 leaves one dense at the end which is needed for the targets (when using advancedmodel3 or 4 [see build_model])
#              layer.trainable=False  
       loaded_model.build(input_shape=(X_train.shape[1],X_train.shape[2]))
       loaded_model_functional_api = loaded_model.model 
       Pfad4 = loaded_model_functional_api(intro_out) 

       
       loaded_model=ModelLaden(ASISModel.destination,ASISModel.modelname)
       loaded_model.load_weights(ASISModel.destination+ASISModel.modelweights+'.h5') 
       loaded_model.trainable = False
#       for layer in loaded_model.layers[1:6]:  #0-6 is all 0:4 leaves one dense at the end which is needed for the targets (when using advancedmodel3 or 4 [see build_model])
#              layer.trainable=False         
       Pfad5 = loaded_model(intro_out) 
       
       i = layers.concatenate([Pfad1, Pfad2, Pfad3, Pfad4, Pfad5])

#       Outro_out=layers.Bidirectional(GRU(Var.hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(i)                   
#       Outro_out = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(Outro_out)
       Outro_out = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(i)

       model = Model(inputs=inp,outputs=Outro_out)  
       
       return model 

#%%      
def Transfer_wide_Beta_GRU_2(X_train,Y_train,Var):
       # HERE WE BUILD THE MODEL PATH AND LOAD THE WEIGTHS INTO EACH LAYER. THEREBY WE CAN DETERMINE WHICH LAYER SHOULD BE FIXED AND WHICH RETRAINED

       class ASQSModel:
              destination=('C:/Users/310122653/Documents/GitHub/DNN/Results/')
              modelname='4_Create_weighs'
              modelweights='4_Create_weighs_weigths'  
       
       class ASISModel:
              destination=('C:/Users/310122653/Documents/GitHub/DNN/Results/')
              modelname='4_Create_weighs'
              modelweights='4_Create_weighs_weigths'     
       
       class ASCTWModel:
              destination=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/')
              modelname=''
              modelweights=''  
       
       class QSISModel:
              destination=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/')
              modelname=''
              modelweights=''          
       
       class QSCTWModel:
              destination=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/')
              modelname=''
              modelweights=''
       
       class ISCTWModel:
              destination=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/')
              modelname=''
              modelweights=''          

           
       def ModelLaden(destination,modelname):
              json_file = open(destination+modelname+'.json', 'r')
              loaded_model_json = json_file.read()
              json_file.close()
              loaded_model = model_from_json(loaded_model_json)
              # load weights into new model
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
       
       
       def Block_unit_transfer(Var,hidden_units, Bimodel):
            layer_weights=list()  
            loaded_model=ModelLaden(Bimodel.destination,Bimodel.modelname)
            loaded_model.load_weights(Bimodel.destination+Bimodel.modelweights+'.h5') 
            for layer in loaded_model.layers:
                   layer_weights.append( layer.get_weights())
                   # list of numpy arrays

            def unit_transfer(x):
#                 ident = x
                 x=Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.),weights=layer_weights[2],trainable=False,)(x)                 
                 x=layers.Bidirectional(GRU(hidden_units, activation=Var.activationF, return_sequences=True, 
                                weights=layer_weights[3],   
                                trainable=False,
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(x)
                 x=layers.Bidirectional(GRU(hidden_units, return_sequences=True,
                                weights=layer_weights[4],
                                trainable=False,                                            
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(x)                                      
                 x=Bidirectional(LSTM(Var.hidden_units, return_sequences=True,
                                weights=layer_weights[5],
                                trainable=False,                                           
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout))(x)  
                 x=Bidirectional(LSTM(Var.hidden_units, return_sequences=True,
                                weights=layer_weights[6],
                                trainable=False,                                           
                                kernel_regularizer=regularizers.l2(Var.Kr),
                                activity_regularizer=regularizers.l2(Var.Ar),
                                kernel_constraint=max_norm(max_value=3.), 
                                dropout=Var.dropout, recurrent_dropout=Var.dropout))(x)   
                 x=layers.Dropout(Var.dropout, noise_shape=(None, 1, hidden_units*2))(x) 
                 x=layers.Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.))(x)                                                  
#                 x=layers.add([ident,x]) 
                 return x
            return unit_transfer                           


       inp = Input(shape=(X_train.shape[1],X_train.shape[2]))
#       i = inp
       i=layers.Masking(mask_value=666,input_shape=(X_train.shape[1],X_train.shape[2]))(inp)
       i=layers.Dropout(Var.dropout/2, noise_shape=(None, 1, X_train.shape[2]))(i)
       ii=layers.Dense(Var.Dense_Unit, activation=Var.activationF, kernel_constraint=max_norm(max_value=3.))(i) 
       intro_out=BatchNormalization(axis=1)(ii)
            
       Pfad1 = cake(Var,32)(intro_out) 
       Pfad1 = cake(Var,32)(Pfad1) 
       
       Pfad2 = cake(Var,2)(intro_out) 
       Pfad2 = cake(Var,2)(Pfad2) 

       Pfad3 = cake(Var,64)(intro_out) 
       Pfad3 = cake(Var,64)(Pfad3) 
       
       Pfad4 = Block_unit_transfer(Var,Var.hidden_units,ASQSModel)(i)         
       Pfad5 = Block_unit_transfer(Var,Var.hidden_units,ASISModel)(i) 
       
       i = layers.concatenate([Pfad1, Pfad2, Pfad3, Pfad4, Pfad5])

#       Outro_out=layers.Bidirectional(GRU(Var.hidden_units, return_sequences=True, kernel_constraint=max_norm(max_value=3.), dropout=Var.dropout, recurrent_dropout=Var.dropout))(i)                   
#       Outro_out = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(Outro_out)
       Outro_out = Dense(Y_train.shape[-1],activation='softmax', kernel_constraint=max_norm(max_value=3.))(i)

       model = Model(inputs=inp,outputs=Outro_out)  
       
       return model 