#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:04:31 2018

@author: 310122653
"""

from build_model import basic_dense_model
from build_model import LSTM_model_1
from build_model import LSTM_model_1_gen
from build_model import LSTM_model_2
from build_model import model_3_LSTM
from build_model import model_3_LSTM_advanced
from build_model import model_3b_LSTM_advanced

from build_model import model_3_LSTM_advanced_seq
from build_model import model_3_LSTM_advanced_no_bi
from build_model import model_4_GRU
from build_model import model_4_GRU_advanced

from build_model_bi import model_4_GRU_advanced1
from build_model_bi import model_4_GRU_advanced2
from build_model_bi import model_4_GRU_advanced3
from build_model_bi import model_4_GRU_advanced4
from build_model_bi import model_4_GRU_advanced5
from build_model_bi import model_4_GRU_advanced6
from build_model_bi import model_4_GRU_advanced7
from build_model_bi2 import model_3_LSTM_advanced1
from build_model_bi2 import model_3_LSTM_advanced2
from build_model_bi2 import model_3_LSTM_advanced3
from build_model_bi2 import model_3_LSTM_advanced4
from build_model_bi2 import model_3_LSTM_advanced5
from build_model_bi2 import model_3_LSTM_advanced6
from build_model_bi2 import model_3_LSTM_advanced7

from build_model_residual import ResNet_deep_Beta_LSTM
from build_model_residual import ResNet_deep_Beta_GRU
from build_model_residual import ResNet_wide_Beta_LSTM
from build_model_residual import ResNet_wide_Beta_GRU
from build_model_residual import ResNet_deep_Beta_GRU_growing

from build_model_transfer import Transfer_wide_Beta_GRU
from build_model_transfer import Transfer_wide_Beta_GRU_2

def whichmodel(Var,X_train,Y_train):
    if Var.model=='model_3_LSTM_advanced':
           model=model_3_LSTM_advanced(X_train,Y_train,Var)
    if Var.model=='model_3b_LSTM_advanced':
           model=model_3b_LSTM_advanced(X_train,Y_train,Var)           
    if Var.model=='model_3_LSTM_advanced_seq':
           model=model_3_LSTM_advanced(X_train,Y_train,Var)
    if Var.model=='model_3__LSTM_advanced_no_bi':
           model=model_3__LSTM_advanced_no_bi(X_train,Y_train,Var)
    if Var.model=='model_4_GRU':
           model=model_4_GRU(X_train,Y_train,Var)
    if Var.model=='model_4_GRU_advanced':
           model=model_4_GRU_advanced(X_train,Y_train,Var)   
           
    if Var.model=='model_4_GRU_advanced1':
           model=model_4_GRU_advanced1(X_train,Y_train,Var) 
    if Var.model=='model_4_GRU_advanced2':
           model=model_4_GRU_advanced2(X_train,Y_train,Var)      
    if Var.model=='model_4_GRU_advanced3':
           model=model_4_GRU_advanced3(X_train,Y_train,Var)   
    if Var.model=='model_4_GRU_advanced4':
           model=model_4_GRU_advanced4(X_train,Y_train,Var)       
    if Var.model=='model_4_GRU_advanced5':
           model=model_4_GRU_advanced5(X_train,Y_train,Var)   
    if Var.model=='model_4_GRU_advanced6':
           model=model_4_GRU_advanced6(X_train,Y_train,Var)   
    if Var.model=='model_4_GRU_advanced7':
           model=model_4_GRU_advanced7(X_train,Y_train,Var)              
    if Var.model=='model_3_LSTM_advanced1':
           model=model_3_LSTM_advanced1(X_train,Y_train,Var)
    if Var.model=='model_3_LSTM_advanced2':
           model=model_3_LSTM_advanced2(X_train,Y_train,Var)
    if Var.model=='model_3_LSTM_advanced3':
           model=model_3_LSTM_advanced3(X_train,Y_train,Var)
    if Var.model=='model_3_LSTM_advanced4':
           model=model_3_LSTM_advanced4(X_train,Y_train,Var)
    if Var.model=='model_3_LSTM_advanced5':
           model=model_3_LSTM_advanced5(X_train,Y_train,Var)
    if Var.model=='model_3_LSTM_advanced6':
           model=model_3_LSTM_advanced6(X_train,Y_train,Var)
    if Var.model=='model_3_LSTM_advanced7':
           model=model_3_LSTM_advanced7(X_train,Y_train,Var)

           
    if Var.model=='ResNet_deep_Beta_LSTM':
           model=ResNet_deep_Beta_LSTM(X_train,Y_train,Var)    
    if Var.model=='ResNet_deep_Beta_GRU':
           model=ResNet_deep_Beta_GRU(X_train,Y_train,Var)             
    if Var.model=='ResNet_wide_Beta_LSTM':
           model=ResNet_wide_Beta_LSTM(X_train,Y_train,Var)             
    if Var.model=='ResNet_wide_Beta_GRU':
           model=ResNet_wide_Beta_GRU(X_train,Y_train,Var)  
    if Var.model=='ResNet_deep_Beta_GRU_growing':
           model=ResNet_deep_Beta_GRU_growing(X_train,Y_train,Var)             
     
    if Var.model=='Transfer_wide_Beta_GRU':
           model=Transfer_wide_Beta_GRU(X_train,Y_train,Var)   
    if Var.model=='Transfer_wide_Beta_GRU_2':
           model=Transfer_wide_Beta_GRU_2(X_train,Y_train,Var) 
           
    return model       