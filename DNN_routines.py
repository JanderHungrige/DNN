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

#from Use_imbalanced_learn import cmplx_Oversampling


import keras
from keras import optimizers 
from keras import losses
from keras import metrics
from keras import models
from keras import layers
from keras import callbacks 

import keras.backend as K

from keras.utils import np_utils
from itertools import product
from functools import partial

from build_model import basic_dense_model
from build_model import LSTM_model_1
from build_model import LSTM_model_1_gen
from build_model import LSTM_model_2
from build_model import model_3_LSTM
from build_model import model_3_LSTM_advanced
from build_model import model_3_LSTM_advanced_seq
from build_model import model_3_LSTM_advanced_no_bi
from build_model import model_4_GRU
from build_model import model_4_GRU_advanced


from build_model_residual import ResNet_deep_Beta_LSTM
from build_model_residual import ResNet_wide_Beta_LSTM
from build_model_residual import ResNet_wide_Beta_GRU

from build_model_transfer import Transfer_wide_Beta_GRU
from build_model_transfer import Transfer_wide_Beta_GRU_2


from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from Performance_callback import categorical_accuracy_no_mask
from Performance_callback import f1_precicion_recall_acc
from Performance_callback import f1_prec_rec_acc_noMasking
#import __main__  


import pdb# use pdb.set_trace() as breakpoint

#%%
def KeraS(X_train, Y_train, X_val, Y_val, X_test, Y_test, Var):
       
#selecte_babies are the babies without test baby
#### CREATING THE sampleweight FOR SELECTED BABIES  
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
    F1_all_collect=[];K_collect=[]
    all_test_metric=[];all_test_loss=[];all_train_metric=[];all_train_loss=[];all_val_metric=[];all_val_loss=[]
    resultsK=[];mean_test_metric=[];mean_train_metric=[]
    all_val_f1=[];all_val_recall=[];all_val_precisions=[];all_val_no_mask_acc=[]
    all_train_f1=[];all_train_recall=[];all_train_precisions=[];all_train_no_mask_acc=[]
              
    
#BUILT MODEL    
    if Var.model=='model_3_LSTM_advanced':
           model=model_3_LSTM_advanced(X_train,Y_train,Var)
    if Var.model=='model_3_LSTM_advanced_seq':
           model=model_3_LSTM_advanced(X_train,Y_train,Var)
    if Var.model=='model_3__LSTM_advanced_no_bi':
           model=model_3__LSTM_advanced_no_bi(X_train,Y_train,Var)
    if Var.model=='model_4_GRU':
           model=model_4_GRU(X_train,Y_train,Var)
    if Var.model=='model_4_GRU_advanced':
           model=model_4_GRU_advanced(X_train,Y_train,Var)   
           
    if Var.model=='ResNet_deep_Beta_LSTM':
           model=ResNet_deep_Beta_LSTM(X_train,Y_train,Var)           
    if Var.model=='ResNet_wide_Beta_LSTM':
           model=ResNet_wide_Beta_LSTM(X_train,Y_train,Var)             
    if Var.model=='ResNet_wide_Beta_GRU':
           model=ResNet_wide_Beta_GRU(X_train,Y_train,Var)  
     
    if Var.model=='Transfer_wide_Beta_GRU':
           model=Transfer_wide_Beta_GRU(X_train,Y_train,Var)   
    if Var.model=='Transfer_wide_Beta_GRU_2':
           model=Transfer_wide_Beta_GRU_2(X_train,Y_train,Var)               

    if Var.usedPC=='Philips': # Plotting model
           from keras.utils.vis_utils import plot_model    
           plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True) 
           
    tensorboard = TensorBoard(log_dir='./Results/logs') 

    checkp = ModelCheckpoint(filepath='./Results/'+Var.runningNumber+'_'+Var.description+'_checkpointbestmodel.hdf5',  
                                   monitor='val_loss', 
                                   verbose=1, 
                                   save_best_only=True, 
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1) 
       
    early_stopping_callback = EarlyStopping(monitor='val_categorical_accuracy',
                                            min_delta=0.001, 
                                            patience=Var.early_stopping_patience, 
                                            verbose=0, 
                                            mode='auto')#from build_model_residual import ResNet_LSTM_1  

    #https://github.com/keras-team/keras/issues/2115#issuecomment-315571824    
    class WeightedCategoricalCrossEntropy(object):

      def __init__(self, weights):
             nb_cl = len(weights)
             self.weights = np.ones((nb_cl, nb_cl))
             for class_idx, class_weight in weights.items():
                    self.weights[0][class_idx] = class_weight
                    self.weights[class_idx][0] = class_weight
             self.__name__ = 'w_categorical_crossentropy'

      def __call__(self, y_true, y_pred):
             return self.w_categorical_crossentropy(y_true, y_pred)

      def w_categorical_crossentropy(self, y_true, y_pred):
             nb_cl = len(self.weights)
             final_mask = K.zeros_like(y_pred[..., 0])
             y_pred_max = K.max(y_pred, axis=-1)
             y_pred_max = K.expand_dims(y_pred_max, axis=-1)
             y_pred_max_mat = K.equal(y_pred, y_pred_max)
             for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
                    w = K.cast(self.weights[c_t, c_p], K.floatx())
                    y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
                    y_t = K.cast(y_true[..., c_t], K.floatx())
                    final_mask += w * y_p * y_t
             return K.categorical_crossentropy(y_true,y_pred) * final_mask

      
   # calculating the wight matrix for the classes which will be used by the weighted loss function  
#    weights_per_class=list(Var.weight_dict.values())
#    #https://github.com/keras-team/keras/issues/2115
#    #https://github.com/keras-team/keras/issues/2115#issuecomment-315571824
#    weight_Matrix = np.ones((len(Var.label),len(Var.label)))
#    for i in range(len(Var.label)):
#           for j in range(len(Var.label)):
#                  weight_Matrix[i][j]=weights_per_class[i]/weights_per_class[j]
#                  
#    custom_loss  = partial(w_categorical_crossentropy,weights=weight_Matrix)               
         
#    ncce = partial(w_categorical_crossentropy, weights=Var.weight_dict)
#    custom_loss.__name__ ='w_categorical_crossentropy'    
    
#MODEL PARAMETERS   
   
    Var.weight_dict2=dict() # need to rename keys of the dict to be used in weighted loss function
    for i,j in zip(range(len(Var.label)), Var.weight_dict):
           Var.weight_dict2[i] = Var.weight_dict[j]

#    adam = keras.optimizers.Adam(lr=Var.learning_rate, decay=Var.learning_rate_decay)          
    adam = keras.optimizers.Adam()
#    if Var.Loss_Function=='Weighted_cat_crossentropy' :
#           lossf=WeightedCategoricalCrossEntropy(Var.weight_dict2)
#    else: 
#           lossf=Var.Loss_Function

    model.compile(loss=WeightedCategoricalCrossEntropy(Var.weight_dict2), 
                  optimizer=adam,
                  metrics=Var.Perf_Metric,
                  sample_weight_mode="temporal")    

    callbackmetric=f1_prec_rec_acc_noMasking()
    model.X_train_jan = X_train
    model.Y_train_jan = Y_train
    model.label=Var.label
    model.Jmethod=Var.Jmethod
    model.train_f1=[];model.train_precision=[];model.train_recall=[];model.train_accuracy=[];model.train_accuracy=[]
    model.val_f1=[];model.val_precision=[];model.val_recall=[];model.val_accuracy=[];model.val_accuracy=[]
    
# TRAIN MODEL (in silent mode, verbose=0)       
    history=model.fit(x=X_train,
                      y=Y_train,
                      verbose=1,
                      epochs=Var.Epochs,
                      batch_size=Var.batchsize,
                      sample_weight=Var.class_weights,
                      validation_data=(X_val,Y_val),                       
                      shuffle=True,
                      callbacks=[checkp,callbackmetric])

    print(model.summary()) 
#EVALUATE MODEL     
    test_loss,test_metric=model.evaluate(X_test,Y_test,batch_size=Var.batchsize)        
    prediction = model.predict(X_test, batch_size=Var.batchsize) 
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
    resultsK.append(cohen_kappa_score(Y_test_Result.ravel(),prediction_base,labels=Var.label))        

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
    
    all_val_f1.append(model.val_f1)
    all_val_recall.append(model.val_recall)
    all_val_precisions.append(model.val_precision)
    all_val_no_mask_acc.append(model.val_accuracy)
    all_train_f1.append(model.val_f1)
    all_train_recall.append(model.val_recall)
    all_train_precisions.append(model.val_precision)
    all_train_no_mask_acc.append(model.val_accuracy)
     
       
#    mean_test_metric=np.mean(all_test_metric) # Kappa is not calculated per epoch but just per fold. Therefor we generate on mean Kappa
#    mean_train_metric=np.mean(all_train_metric,axis=0)
    mean_test_metric=all_test_metric # Kappa is not calculated per epoch but just per fold. Therefor we generate on mean Kappa
    mean_train_metric= np.mean(all_train_metric,axis=0)
    mean_val_metric=   np.mean(all_val_metric,axis=0)    
    mean_test_loss=    np.mean(all_test_loss,axis=0)    
    mean_train_loss=   np.mean(all_train_loss,axis=0)
    mean_val_loss=     np.mean(all_val_loss,axis=0)      
    mean_k=            np.mean(resultsK)
    
    mean_val_f1=np.mean(all_val_f1)
    mean_val_recall=np.mean(all_val_recall)
    mean_val_precicion=np.mean(all_val_precisions)
    mean_val_no_mask_acc=np.mean(all_val_no_mask_acc)
    
    mean_train_f1=np.mean(all_train_f1)
    mean_train_recall=np.mean(all_train_recall)
    mean_train_precicion=np.mean(all_train_precisions)
    mean_train_no_mask_acc=np.mean(all_train_no_mask_acc)
#    from sklearn.metrics import classification_report
#    target_names = ['AS', 'QS', 'CTW','IS']
#    Report=(classification_report(Y_test_Result.ravel(), prediction_base, target_names=target_names))
#
#      
    model.perfmatrix= callbackmetric
    return model, resultsK, mean_k, mean_train_metric, mean_val_metric, mean_train_loss, mean_val_loss, mean_test_metric, mean_test_loss,\
    mean_val_f1,mean_val_recall,mean_val_precicion,mean_val_no_mask_acc,mean_train_f1,mean_train_recall,mean_train_precicion,mean_train_no_mask_acc

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
    mean_test_loss=np.mean(all_test_loss,axis=0)    
    mean_train_loss=np.mean(all_train_loss,axis=0)
    mean_val_loss=np.mean(all_val_loss,axis=0)      
    mean_k=np.mean(resultsK)
      
    
    return resultsK, mean_k, mean_train_metric, mean_val_metric, mean_train_loss, mean_val_loss, mean_test_metric, mean_test_loss
#%%