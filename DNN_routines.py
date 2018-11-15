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

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from Performance_callback import categorical_accuracy_no_mask
from Performance_callback import f1_precicion_recall_acc
from Performance_callback import f1_prec_rec_acc_noMasking
#import __main__  

from select_model import whichmodel
import pdb# use pdb.set_trace() as breakpoint

#%%
def KeraS(X_train, Y_train, X_val, Y_val, X_test, Y_test, Var):
       
#selecte_babies are the babies without test baby
#### CREATING THE sampleweight FOR SELECTED BABIES  
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)            
    
#BUILT MODEL    
    model=whichmodel(Var,X_train,Y_train)        

    if Var.usedPC=='Philips': # Plotting model
           from keras.utils.vis_utils import plot_model    
           plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True) 
           plot_model(model, to_file='model_plot.svg', show_shapes=True, show_layer_names=True) 
           
    tensorboard = TensorBoard(log_dir=Var.resultpath +'/tensorboardlogs/'+Var.description,
                              histogram_freq=1,write_graph=True, write_images=False) 

    checkp = ModelCheckpoint(filepath=Var.resultpath + '/'+Var.runningNumber+'_'+Var.description+'Fold_'+str(Var.Fold)+'_checkpointbestmodel.hdf5',  
                                   monitor='val_loss', 
                                   verbose=0, 
                                   save_best_only=True, 
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1) 
       
    early_stopping_callback = EarlyStopping(monitor='val_categorical_accuracy',
                                            min_delta=0.01, 
                                            patience=Var.early_stopping_patience, 
                                            verbose=0, 
                                            mode='auto')#from build_model_residual import ResNet_LSTM_1  

    #https://github.com/keras-team/keras/issues/2115#issuecomment-315571824    
    class WeightedCategoricalCrossEntropy(object):

      def __init__(self, weights):
          if np.ndim(weights)==0:
             nb_cl = len(weights)
             self.weights = np.ones((nb_cl, nb_cl))
             for class_idx, class_weight in weights.items():
                    self.weights[0][class_idx] = class_weight
                    self.weights[class_idx][0] = class_weight
          elif np.ndim(weights)>1: # if the dimesion of the input dimesnion isnot just an array, we give it a full matrix with weights. OTherwise we give it jsut class weights
             self.weights=weights
              
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
#    #https://github.com/keras-team/keras/issues/2115
#    #https://github.com/keras-team/keras/issues/2115#issuecomment-315571824   
    weights_per_class=list(Var.weight_dict.values())
    weight_Matrix = np.ones((len(Var.label),len(Var.label)))
    for i in range(len(Var.label)):
           for j in range(len(Var.label)):
                  weight_Matrix[i][j]=weights_per_class[i]/weights_per_class[j] 
    
#MODEL PARAMETERS   
   
    Var.weight_dict2=dict() # need to rename keys of the dict to be used in weighted loss function
    for i,j in zip(range(len(Var.label)), Var.weight_dict):
           Var.weight_dict2[i] = Var.weight_dict[j]

#    adam = keras.optimizers.Adam(lr=Var.learning_rate, decay=Var.learning_rate_decay)          


    if Var.Loss_Function=='Weighted_cat_crossentropy1':
        model.compile(loss=WeightedCategoricalCrossEntropy(Var.weight_dict2),        
                      optimizer=Var.optimizer,
                      metrics=Var.Perf_Metric,
                      sample_weight_mode="temporal") 
    elif Var.Loss_Function=='Weighted_cat_crossentropy2':
        model.compile(loss=WeightedCategoricalCrossEntropy(weight_Matrix),        
                      optimizer=Var.optimizer,
                      metrics=Var.Perf_Metric,
                      sample_weight_mode="temporal")       
    elif Var.Loss_Function=='categorical_crossentropy':
        model.compile(loss='categorical_crossentropy',                               
                  optimizer=Var.optimizer,
                  metrics=Var.Perf_Metric,
                  sample_weight_mode="temporal")
    else: 
        import sys
        sys.exit(print(Var.Loss_Function + ' Seems wrong spelling')) 

    callbackmetric=f1_prec_rec_acc_noMasking()
    model.X_train_jan = X_train
    model.Y_train_jan = Y_train
    model.label=Var.label
    model.Jmethod=Var.Jmethod
#    model.train_f1=[];model.train_k=[];model.train_precision=[];model.train_recall=[];model.train_accuracy=[];model.train_accuracy=[]
#    model.val_f1=[];model.val_k=[];model.val_precision=[];model.val_recall=[];model.val_accuracy=[];model.val_accuracy=[]
    
# TRAIN MODEL (in silent mode, verbose=0)       
    history=model.fit(x=X_train,
                      y=Y_train,
                      verbose=1,
                      epochs=Var.Epochs,
                      batch_size=Var.batchsize,
                      sample_weight=Var.class_weights,
                      validation_data=(X_val,Y_val),                       
                      shuffle=True,
                      callbacks=[callbackmetric])

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
# Here we save the results from the callback into a class as the callback itself makes problems with saving with pickle
    callbackmetric.train_metric.append(history.history['categorical_accuracy'])
    callbackmetric.train_loss.append(history.history['loss'])
    callbackmetric.val_metric.append(history.history['val_categorical_accuracy'] )
    callbackmetric.val_loss.append(history.history['val_loss'] )   
    
#    from sklearn.metrics import classification_report
#    target_names = ['AS', 'QS', 'CTW','IS']
#    Report=(classification_report(Y_test_Result.ravel(), prediction_base, target_names=target_names))
#
#    Ergebniss=callbackmetric()
    model.perfmatrix=callbackmetric
    return model,callbackmetric
