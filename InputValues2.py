#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 21:28:12 2018

@author: 310122653
"""

          
import numpy as np
import keras
from keras import optimizers 

def inputcombinations2(whatID):
    adam = keras.optimizers.Adam()
    nadam=keras.optimizers.Nadam()
    adamax=keras.optimizers.Adamax()
    adaDelta=keras.optimizers.Adadelta()
    rmsProp=keras.optimizers.RMSprop()
    
    if whatID in np.arange(300,307):
        model='ResNet_wide_Beta_GRU'
        Loss_Function='Weighted_cat_crossentropy2'#Weighted_cat_crossentropy or categorical_crossentropy OR mean_squared_error IF BINARY : binary_crossentropy
        label=[1,2,3,4,6]       
        Optimizer=adam
        
    if whatID in np.arange(400,442):
        model='ResNet_wide_Beta_GRU'
        Loss_Function='Weighted_cat_crossentropy2'
        Optimizer=adam
        
    if whatID in np.arange(600,607):
        model='ResNet_deep_Beta_GRU'
        Loss_Function='categorical_crossentropy'#Weighted_cat_crossentropy or categorical_crossentropy OR mean_squared_error IF BINARY : binary_crossentropy
        label=[1,2,3,4,6]       
        Optimizer=adam
        
    if whatID in np.arange(700,742):
        model='ResNet_deep_Beta_GRU'  #ResNet_deep_Beta_GRU  categorical_crossentropy
        Loss_Function='categorical_crossentropy'
        Optimizer=adam        
#%% Transferleraning    
    if whatID in [500]:
        description='Transfer_ECG_InSe' # Bi_ASQS  Bi_ASIS  Bi_ASCTW  Bi_QSIS  Bi_QSCTW  Bi_ISCTW
        Loss_Function='categorical_crossentropy'#Weighted_cat_crossentropy or categorical_crossentropy OR mean_squared_error IF BINARY : binary_crossentropy

    if whatID in [501]:
        description='Transfer_ECG_InSe' # Bi_ASQS  Bi_ASIS  Bi_ASCTW  Bi_QSIS  Bi_QSCTW  Bi_ISCTW      
        Loss_Function='Weighted_cat_crossentropy1'#Weighted_cat_crossentropy or categorical_crossentropy OR mean_squared_error IF BINARY : binary_crossentropy
    if whatID in [502]:
        description='Transfer_ECG_InSe' # Bi_ASQS  Bi_ASIS  Bi_ASCTW  Bi_QSIS  Bi_QSCTW  Bi_ISCTW      
        Loss_Function='Weighted_cat_crossentropy2'#Weighted_cat_crossentropy or categorical_crossentropy OR mean_squared_error IF BINARY : binary_crossentropy
              
    if whatID in [500,501,502]:
        label=[1,2,3,4,6]
        dataset='ECG+InSe' #MMC+ECG+InSe     MMC+ECG   MMC+InSe   ECG+InSe   MMC   ECG   InSe   
        Optimizer=adam
        model='Transfer_wide_Beta_GRU_2'
        
#%% Residual on all 300-306
    if whatID in [300,600]:
        description='Res_MMC' 
        dataset='MMC' #MMC+ECG+InSe     MMC+ECG   MMC+InSe   ECG+InSe   MMC   ECG   InSe   

    if whatID in [301,601]:
        description='Res_ECG'
        dataset='ECG' #MMC+ECG+InSe     MMC+ECG   MMC+InSe   ECG+InSe   MMC   ECG   InSe   

    if whatID in [302,602]:
        description='Res_InSe' 
        dataset='InSe' #MMC+ECG+InSe     MMC+ECG   MMC+InSe   ECG+InSe   MMC   ECG   InSe   
       
    if whatID in [303,603]:
        description='Res_MMC_InSe' 
        dataset='MMC+InSe' #MMC+ECG+InSe     MMC+ECG   MMC+InSe   ECG+InSe   MMC   ECG   InSe   
               
    if whatID in [304,604]:
        description='Res_ECG_InSe' 
        dataset='ECG+InSe' #MMC+ECG+InSe     MMC+ECG   MMC+InSe   ECG+InSe   MMC   ECG   InSe   
               
    if whatID in [305,605]:
        description='Res_MMC_ECG' 
        dataset='MMC+ECG' #MMC+ECG+InSe     MMC+ECG   MMC+InSe   ECG+InSe   MMC   ECG   InSe   
               
    if whatID in [306,606]:
        description='Res_MMC_ECG_InSe' 
        dataset='MMC+ECG+InSe' #MMC+ECG+InSe     MMC+ECG   MMC+InSe   ECG+InSe   MMC   ECG   InSe   
        

        
#%% Residual on bi-states 400-441
        
    if whatID in np.arange(400,437,6) or whatID in np.arange(700,737,6):
       description='Res_Bi_ASQS' # Bi_ASQS  Bi_ASIS  Bi_ASCTW  Bi_QSIS  Bi_QSCTW  Bi_ISCTW
       label=[1,2]       
    if whatID in np.arange(401,438,6) or whatID in np.arange(701,738,6):
       description='Res_Bi_ASIS'
       label=[1,6]       
    if whatID in np.arange(402,439,6) or whatID in np.arange(702,739,6):        
       description='Res_Bi_ASCTW' 
       label=[1,3,4]       
    if whatID in np.arange(403,440,6) or whatID in np.arange(703,740,6):          
       description='Res_Bi_QSIS' 
       label=[2,6]       
    if whatID in np.arange(404,441,6) or whatID in np.arange(704,741,6): 
       description='Res_Bi_QSCTW' 
       label=[2,3,4]       
    if whatID in np.arange(405,442,6) or whatID in np.arange(705,742,6): 
       description='Res_Bi_ISCTW'    
       label=[3,4,6]        
 
    if whatID in np.arange(400,406) or whatID in np.arange(700,706):
       dataset='MMC+ECG+InSe' #MMC+ECG+InSe     MMC+ECG   MMC+InSe   ECG+InSe   MMC   ECG   InSe         "cECG"   
    if whatID in np.arange(406,412) or whatID in np.arange(706,712):
       dataset='MMC+ECG'   
    if whatID in np.arange(412,418) or whatID in np.arange(712,718):
       dataset='MMC+InSe'
    if whatID in np.arange(418,424) or whatID in np.arange(718,724):
       dataset='ECG+InSe'
    if whatID in np.arange(424,430) or whatID in np.arange(724,730):
       dataset='MMC'
    if whatID in np.arange(430,436) or whatID in np.arange(730,736):
       dataset='ECG'
    if whatID in np.arange(436,442) or whatID in np.arange(736,742):
       dataset='InSe'
       
        

       
    return description, dataset, model, label,Loss_Function,Optimizer    