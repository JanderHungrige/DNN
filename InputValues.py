#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:35:02 2018

@author: 310122653
"""
import numpy as np
def inputcombinations(whatID):
    
    if whatID in np.arange(39,118,6) or whatID in np.arange(123,202,6):
       description='Bi_ASQS' # Bi_ASQS  Bi_ASIS  Bi_ASCTW  Bi_QSIS  Bi_QSCTW  Bi_ISCTW
    if whatID in np.arange(40,119,6) or whatID in np.arange(124,203,6):
       description='Bi_ASIS' # Bi_ASQS  Bi_ASIS  Bi_ASCTW  Bi_QSIS  Bi_QSCTW  Bi_ISCTW
    if whatID in np.arange(41,120,6) or whatID in np.arange(125,204,6):        
       description='Bi_ASCTW' # Bi_ASQS  Bi_ASIS  Bi_ASCTW  Bi_QSIS  Bi_QSCTW  Bi_ISCTW
    if whatID in np.arange(42,121,6) or whatID in np.arange(126,205,6):          
       description='Bi_QSIS' # Bi_ASQS  Bi_ASIS  Bi_ASCTW  Bi_QSIS  Bi_QSCTW  Bi_ISCTW
    if whatID in np.arange(43,122,6) or whatID in np.arange(127,206,6): 
       description='Bi_QSCTW' # Bi_ASQS  Bi_ASIS  Bi_ASCTW  Bi_QSIS  Bi_QSCTW  Bi_ISCTW
    if whatID in np.arange(44,123,6) or whatID in np.arange(128,207,6): 
        description='Bi_ISCTW' # Bi_ASQS  Bi_ASIS  Bi_ASCTW  Bi_QSIS  Bi_QSCTW  Bi_ISCTW     
       
 
    if whatID in np.arange(39,51) or whatID in np.arange(123,135):
       dataset='MMC+ECG+InSe' #MMC+ECG+InSe     MMC+ECG   MMC+InSe   ECG+InSe   MMC   ECG   InSe         "cECG"   
    if whatID in np.arange(51,63) or whatID in np.arange(135,147):
       dataset='MMC+ECG'   
    if whatID in np.arange(63,75) or whatID in np.arange(147,159):
       dataset='MMC+InSe'
    if whatID in np.arange(75,87) or whatID in np.arange(159,171):
       dataset='ECG+InSe'
    if whatID in np.arange(87,99) or whatID in np.arange(171,183):
       dataset='MMC'
    if whatID in np.arange(99,111) or whatID in np.arange(183,195):
       dataset='ECG'
    if whatID in np.arange(111,123) or whatID in np.arange(195,207):
       dataset='ECG'
       
    if whatID in np.arange(39,44)   or whatID in np.arange(123,129)\
    or whatID in np.arange(51,57)   or whatID in np.arange(135,141)\
    or whatID in np.arange(63,69)   or whatID in np.arange(147,153)\
    or whatID in np.arange(75,81)   or whatID in np.arange(159,165)\
    or whatID in np.arange(87,93)   or whatID in np.arange(171,177)\
    or whatID in np.arange(99,105)  or whatID in np.arange(183,189)\
    or whatID in np.arange(111,117) or whatID in np.arange(195,201)\
    :
       model='model_3_LSTM_advanced' # check DNN_routines KeraS for options model_4_GRU_advanced  model_3_LSTM_advanced

    if whatID in np.arange(45,51)   or whatID in np.arange(129,135)\
    or whatID in np.arange(57,63)   or whatID in np.arange(141,147)\
    or whatID in np.arange(69,75)   or whatID in np.arange(153,159)\
    or whatID in np.arange(81,87)   or whatID in np.arange(165,171)\
    or whatID in np.arange(93,99)   or whatID in np.arange(177,183)\
    or whatID in np.arange(105,111) or whatID in np.arange(189,195)\
    or whatID in np.arange(117,123) or whatID in np.arange(201,207)\
    :
       model='model_4_GRU_advanced' # check DNN_routines KeraS for options model_4_GRU_advanced  model_3_LSTM_advanced

    if description=='Bi_ASQS':
        label=[1,2]
    elif description=='Bi_ASIS':
        label=[1,6]
    elif description=='Bi_ASCTW':
        label=[1,3,4]
    elif description=='Bi_QSIS':
        label=[2,6]
    elif description=='Bi_QSCTW':
        label=[2,3,4]
    elif description=='Bi_ISCTW':
        label=[3,4,6]  

    if whatID in np.arange(39,123): 
       Loss_Function='Weighted_cat_crossentropy'#Weighted_cat_crossentropy or categorical_crossentropy OR mean_squared_error IF BINARY : binary_crossentropy
    if whatID in np.arange(123,207): 
       Loss_Function='categorical_crossentropy'#Weighted_cat_crossentropy or categorical_crossentropy OR mean_squared_error IF BINARY : binary_crossentropy



    return description, dataset, model, label,Loss_Function