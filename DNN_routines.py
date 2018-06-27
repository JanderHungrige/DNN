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
from sklearn.metrics import precision_score, recall_score

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
from build_model import LSTM_model_1_gen
from build_model import LSTM_model_2
from build_model import LSTM_model_3
from build_model import LSTM_model_3_advanced

from build_model_residual import ResNet_LSTM_1

#import __main__  


import pdb# use pdb.set_trace() as breakpoint

#%%
def KeraS(X_train, Y_train, X_val, Y_val, X_test, Y_test, batchsize,Epochs,dropout,hidden_units,label,class_weights,learning_rate,learning_rate_decay,activationF, Loss_Function, Perf_Metric):
       
#selecte_babies are the babies without test baby
#### CREATING THE sampleweight FOR SELECTED BABIES  
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
    F1_all_collect=[];K_collect=[]
    all_test_metric=[];all_test_loss=[];all_train_metric=[];all_train_loss=[];all_val_metric=[];all_val_loss=[]
    resultsK=[];mean_test_metric=[];mean_train_metric=[]
              

#BUILT MODEL    
#    model=basic_dense_model(X_train,Y_train)
#    model=LSTM_model_1(X_train,Y_train,dropout,hidden_units,MaskWert)
#    model=LSTM_model_2(X_train,Y_train,dropout,hidden_units,MaskWert)
#    model=LSTM_model_3_advanced(X_train,Y_train,dropout,hidden_units,activationF)
    residual_blocks=1
    model=ResNet_LSTM_1(X_train,Y_train,dropout,hidden_units,activationF,residual_blocks)
#MODEL PARAMETERS    
    model.compile(loss=Loss_Function, 
                  optimizer='adam',
                  metrics=Perf_Metric,
                  sample_weight_mode="temporal")       
    model.optimizer.lr=learning_rate #0.0001 to 0.01 default =0.001
    model.optimizer.decay=learning_rate_decay
    

# TRAIN MODEL (in silent mode, verbose=0)       
    history=model.fit(X_train,
                       Y_train,
                       epochs=Epochs,
                       batch_size=batchsize,
                       sample_weight=class_weights,
                       validation_data=(X_val,Y_val),
                       shuffle=False)
    
    from keras.utils.vis_utils import plot_model    
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
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
    
#    from sklearn.metrics import classification_report
#    target_names = ['AS', 'QS', 'CTW','IS']
#    Report=(classification_report(Y_test_Result.ravel(), prediction_base, target_names=target_names))
#
#      
    
    return resultsK, mean_k, mean_train_metric, mean_val_metric, mean_train_loss, mean_val_loss, mean_test_metric, mean_test_loss

#%%
def KeraS_Gen(X_Train_Val_Test,Y_Train_Val_Test,
              steps_per_epoch,Epochs,dropout,hidden_units,label,class_weight,
              lookback):
       
    def create_timesteps(data, time_step):
              for i in range(0, len(data), time_step): #Yield successive n-sized chunks from l
                     yield data[i:i + time_step]

       
    def data_Generator_with_lookback(X_Data,Y_Data,lookback,MaskWert):
           n=0
           if lookback!=1337:
                  X_Data=concatenate(X_Data,axis=0)
                  Y_Data=concatenate(Y_Data,axis=0) 
                  X_Data_lockback=list(create_timesteps(X_Data, lookback))
                  Y_Data_lockback=list(create_timesteps(Y_Data, lookback))                     
                  if X_Data_lockback[-1].shape < X_Data_lockback[-2].shape:  #make the last one the same length as the others, zeropad                       
                             X_Data_lockback[-1]=np.pad(X_Data_lockback[-1],pad_width=((0,X_Data_lockback[-2].shape[0]-X_Data_lockback[-1].shape[0]),(0,0)),mode='constant',constant_values=MaskWert)
                             Y_Data_lockback[-1]=np.pad(Y_Data_lockback[-1],pad_width=((0,Y_Data_lockback[-2].shape[0]-Y_Data_lockback[-1].shape[0]),(0,0)),mode='constant',constant_values=0)
                  pdb.set_trace()              
                  while n<=len(X_Data_lockback): 
                         yield  X_Data_lockback[n], Y_Data_lockback[n]
                         n+=1
                            
           if lookback==1337:
                  while n<=len(X_Data): 
                         yield  X_Data[n], Y_Data[n]
                         n+=1
                                   
       
#selecte_babies are the babies without test baby
#### CREATING THE sampleweight FOR SELECTED BABIES  
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
    F1_all_collect=[];K_collect=[]
    all_test_metric=[];all_test_loss=[];all_train_metric=[];all_train_loss=[];all_val_metric=[];all_val_loss=[]
    resultsK=[];mean_test_metric=[];mean_train_metric=[]
    
    MaskWert=666
              
    X_train=X_Train_Val_Test[0]
    Y_train=Y_Train_Val_Test[0]
     
    X_Val=X_Train_Val_Test[1]
    Y_Val=Y_Train_Val_Test[1] 
    
    Nr_Features=X_train[0].shape[1]
    Nr_labels=Y_train[0].shape[1]
     
#     X_test=X_Train_Val_Test[2]
#     Y_test=Y_Train_Val_Test[2] 
    
# Generators
    training_generator =data_Generator_with_lookback(X_train,Y_train,lookback,MaskWert)
    validation_generator=data_Generator_with_lookback(X_Val,Y_Val,lookback,MaskWert)
    
#BUILT MODEL    
#    model=basic_dense_model(X_train,Y_train)
    model=LSTM_model_1_gen(lookback,Nr_Features,Nr_labels,dropout,hidden_units,MaskWert)
    model.optimizer.lr=0.0001 #0.0001 to 0.01 default =0.001
    model.optimizer.decay=0.0    
    

# TRAIN MODEL (in silent mode, verbose=0)       
    history=model.fit_generator(
                  generator=training_generator,                     
                  epochs=Epochs,
                  steps_per_epoch=steps_per_epoch,
                  class_weight=class_weight,
                  validation_data=validation_generator,
                  validation_steps=steps_per_epoch,
                  shuffle=False)

#EVALUATE MODEL      
    test_loss,test_metric=model.evaluate_generator(data_Generator_with_lookback(X_train,Y_train,lookback,MaskWert),steps=steps_per_epoch)        
    prediction = model.predict_generator(data_Generator_with_lookback(X_train,Y_train,lookback,MaskWert), steps=steps_per_epoch) 
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
#%%