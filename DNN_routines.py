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

from Use_imbalanced_learn import cmplx_Oversampling


import keras
from keras import optimizers 
from keras import losses
from keras import metrics
from keras import models
from keras import layers

from keras.utils import np_utils


import pdb# use pdb.set_trace() as breakpoint

        
def KeraS(X_train_val, y_train_val, X_test, y_test, selected_babies_train, label, fold):
       
#selecte_babies are the babies without test baby
#### CREATING THE sampleweight FOR SELECTED BABIES  
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
    F1_all_collect=[];K_collect=[]

    validation_amount=floor(len(selected_babies_train)/fold) # always one fold validation rest folds training
    if validation_amount <1:
         sys.exit('Error in DNN_routines! choose differnt fold. Validation below 1 patient ')
    if (len(selected_babies_train)/fold) % 1> 0.6: # if we have more than 0.6 as division rest that means that we cut of quite some data at the rest to include in the validation fold
        print('modulu large (>0.6). To lose less of the data for validation use higher or lower fold')     
        
    for Fold in range(fold-1):
        disp(["progressing fold: " ,Fold+1,"of" ,fold])
                      
        a=int(Fold*validation_amount)
        b=int(Fold*validation_amount+validation_amount)

        X_val_per_patient=X_train_val[a:b]
        X_train_per_patient=X_train_val[:] # make a copy of the data not to change the original
        del X_train_per_patient[a:b] #remove the patients which are used for validation 
        

        y_val_per_patient=y_train_val[a:b] 
        y_train_per_patient=y_train_val[:] # make a copy of the data not to change the original
        del y_train_per_patient[a:b] #remove the patients which are used for validation 
                
#creating one big array per               
        X_train=np.vstack(X_train_per_patient)
        y_train=np.vstack(y_train_per_patient)


        if len(X_val_per_patient)>1:
               X_val=np.vstack(X_val_per_patient)
               y_val=np.vstack(y_val_per_patient)

               
        else:
               X_val=X_val_per_patient[0]
               y_val=y_val_per_patient[0]
# One hot encoding of y-lables               
        y_train = np_utils.to_categorical(y_train)  
        y_val = np_utils.to_categorical(y_val)              
        y_test = np_utils.to_categorical(y_test)  
# from 2D to 3D               
        if len(X_train.shape)<3:
            X_train=X_train[:,:,newaxis]               
        if len(X_val.shape)<3:
            X_val=X_val[:,:,newaxis]        
        if len(X_test.shape)<3:
            X_test=X_test[:,:,newaxis]  

        
#        if len(y_train.shape)<3:
#            y_train=y_train[:,:,newaxis]
#            if len(y_train.shape)<3: #if y was only one dimensional before we need to add another axis to gain 3D
#                y_train=y_train[:,:,newaxis]  
#        if len(y_val.shape)<3:
#            y_val=y_val[:,:,newaxis] 
#            if len(y_val.shape)<3:
#                y_val=y_val[:,:,newaxis] 
#        if len(y_test.shape)<3:
#            y_test=y_test[:,:,newaxis]  
#            if len(y_test.shape)<3:
#                y_test=y_test[:,:,newaxis]  
#            
#%% 
        # Build model 
        def build_model():  
            model = models.Sequential()
            model.add(layers.Dense(16, activation='relu',input_shape=(X_train.shape[1],X_train.shape[2])))  
            model.add(layers.Dense(16, activation='relu'))              
            model.add(layers.Dense(y_train.shape[1], activation='softmax'))                      

            model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['mae'])
            return model
     
        model=build_model()
#       # Train the model (in silent mode, verbose=0)       
        history=model.fit(X_train,
                          y_train,
                          epochs=20,
                          batch_size=512,
                          validation_data=(X_val,y_val))

        # Evaluate the model on the validation data        
        val_mse,val_mae=model.evaluate(X_test,y_test,batch_size=512 )
        mae_history=history.history['val_mae_absolute_error']
        all_score.append(val_mae)
        all_mae_histroy.append(mae_history)
        prediction = model.predict(X_test, batch_size=128)    
        resultsK.append(cohen_kappa_score(y_test.ravel(),prediction,labels=label))        
        
    mean_score=np.mean(all_scores)
    mean_k=np.mean(resultsK)
      
        
    
    return resultsK,prediction,mean_score,all_mae_histroy,mean_k 