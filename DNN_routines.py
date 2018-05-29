# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:58:22 2018

@author: 310122653
"""
import itertools
from numpy import *
from pylab import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import cohen_kappa_score

from Use_imbalanced_learn import cmplx_Oversampling


import keras
from keras import optimizers 
from keras import losses
from keras import metrics
from keras import models
from keras import layers
from keras import callbacks 

from keras.utils import np_utils

from build_model import basic_dense_model
from build_model import LSTM_model_1



import pdb# use pdb.set_trace() as breakpoint

def KeraS(X_train, Y_train, X_val, Y_val, X_test, Y_test, batchsize,Epochs,dropout,hidden_units,label):
       
#selecte_babies are the babies without test baby
#### CREATING THE sampleweight FOR SELECTED BABIES  
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
    F1_all_collect=[];K_collect=[]
    all_test_metric=[];all_test_loss=[];all_train_metric=[];all_train_loss=[];all_val_metric=[];all_val_loss=[]
    resultsK=[];mean_test_metric=[];mean_train_metric=[]
              
#%% 
 
        

#BUILT MODEL    
#    model=basic_dense_model(X_train,Y_train)
    model=LSTM_model_1(X_train,Y_train,dropout,hidden_units)


# TRAIN MODEL (in silent mode, verbose=0)       
    history=model.fit(X_train,
                       Y_train,
                       epochs=Epochs,
                       batch_size=batchsize,
                       validation_data=(X_val,Y_val),
                       shuffle=False)

#EVALUATE MODEL      
    test_loss,test_metric=model.evaluate(X_test,Y_test,batch_size=batchsize)        
    prediction = model.predict(X_test, batch_size=batchsize) 
    stackedpred=np.concatenate(prediction, axis=0)
    #make prediction a simple array to match y_train_base      
#    indEx = np.unravel_index(np.argmax(prediction, axis=1), prediction.shape)
    indEx = (np.argmax(stackedpred, axis=1), stackedpred.shape)
    prediction_base=indEx[0]
#    indEy = np.unravel_index(np.argmax(Y_test, axis=1), prediction.shape)
    stackedTest=np.concatenate(Y_test, axis=0)   
    indEy = (np.argmax(stackedTest, axis=1), stackedTest.shape)

    Y_test_Result=indEy[0]   

#       print(history.history.keys())     
    
#COLLECTING RESULTS    
    resultsK.append(cohen_kappa_score(Y_test_Result.ravel(),prediction_base,labels=label))        

    all_test_metric.append(test_metric)
    all_test_loss.append(test_loss)

    train_metric=history.history['categorical_accuracy']
    train_loss = history.history['loss']
    val_metric = history.history['val_categorical_accuracy']         
    val_loss = history.history['val_loss'] 
 
    all_train_metric.append(train_metric)
    all_train_loss.append(train_loss)
    all_val_metric.append(val_metric)
    all_val_loss.append(val_loss)   
    
       
    mean_test_metric=np.mean(all_test_metric) # Kappa is not calculated per epoch but just per fold. Therefor we generate on mean Kappa
    mean_train_metric=np.mean(all_train_metric,axis=0)
    mean_val_metric=np.mean(all_val_metric,axis=0)    
    mean_test_loss=mean(all_test_loss,axis=0)    
    mean_train_loss=np.mean(all_train_loss,axis=0)
    mean_val_loss=np.mean(all_val_loss,axis=0)      
    mean_k=np.mean(resultsK)
      
    
    return resultsK, mean_k, mean_train_metric, mean_val_metric, mean_train_loss, mean_val_loss, mean_test_metric, mean_test_loss