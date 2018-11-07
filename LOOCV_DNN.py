# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:26:01 2017

@author: 310122653
"""
from DNN_routines import KeraS

import itertools
from matplotlib import *
from numpy import *
from pylab import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn import svm, cross_validation
import sys #to add strings together
import pdb # use pdb.set_trace() as breakpoint

from keras.utils import np_utils
from sklearn.utils import class_weight

from collections import deque
import __main__

def leave_one_out_cross_validation(babies,AnnotMatrix_each_patient,FeatureMatrix_each_patient,Var,Varplus,Ergebnisse):
         
       t_a=list()
       classpredictions=list()
       Probabilities=list()
       Performance=list()
       ValidatedPerformance_K=list()
       y_labeled=list()
       classpredictions=list()
       Performance_K=list()
       mean_train_metric=list()
       mean_train_loss=list()
       mean_val_metric=list()
       mean_val_loss=list()
       mean_test_metric=list()
       mean_test_loss=list() 
       mean_val_f1=list()
       mean_val_recall=list()
       mean_val_precicion=list()
       mean_val_no_mask_acc=list()   
       mean_train_f1=list()
       mean_train_recall=list()
       mean_train_precicion=list()
       mean_train_no_mask_acc=list()        
       
       def percentage_split(seq, percentages): #Generator function: Create split based on percentage
           assert sum(percentages) == 1.0
           prv = 0
           size = len(seq)
           cum_percentage = 0
           for p in percentages:
               cum_percentage += p
               nxt = int(cum_percentage * size)
               yield seq[prv:nxt]
               prv = nxt
               
       def create_timesteps(data, time_step):
              for i in range(0, len(data), time_step): #Yield successive n-sized chunks from l
                     yield data[i:i + time_step]
                     
     
       def data_with_lookback(Data_Train_Val_Test,lookback,Wert):
              Data_tmp=concatenate(Data_Train_Val_Test,axis=0)
              Data_Tensor=list(create_timesteps(Data_tmp, lookback))
              if Data_Tensor[-1].shape < Data_Tensor[-2].shape:  #make the last one the same length as the others, zeropad                       
                         Data_Tensor[-1]=np.pad(Data_Tensor[-1],pad_width=((0,Data_Tensor[-2].shape[0]-Data_Tensor[-1].shape[0]),(0,0)),mode='constant',constant_values=Wert)
              Data_Tensor = np.stack((Data_Tensor), axis=0) 
              return Data_Tensor 

       def sample_weight_calc_per_class(y_labeled):
              unique, counts = numpy.unique(concatenate(y_labeled), return_counts=True)	 
              weights=len(concatenate(y_labeled))/counts
              weights= weights/min(weights)#normalize to majority class
              weights_dict=dict(zip(unique.astype(int), weights))
              return weights_dict
       
       def create_sample_Weigth(y_labeled,weights_dict,label):
              results = list()
              for y_mat in y_labeled:
                  weight_matrix = np.zeros(shape = y_mat.shape)
                  for i in label:
                      weight_matrix[y_mat == i] = weights_dict[i]
              
                  results.append(weight_matrix)
              return results
       
       def shiftRbyn(arr, n=0): #circular shift for fold generation
              return arr[n:len(arr):] + arr[0:n:]   
 

#SPLITTING DATA INTO TRAIN+VALIDATION AND TEST SETS
       # Using only the Patients choosen in the wrapper
       FeatureMatrix_for_choosen_patients=[FeatureMatrix_each_patient[k]for k in babies]          # get the feature values for selected babies                  
       AnnotMatrix_for_choosen_patients=[AnnotMatrix_each_patient[k].astype(int) for k in babies]              # get the annotation values for selected babies
           

       # Using only the labels choosen in the wrapper
       idx=[in1d(AnnotMatrix_each_patient[sb],Var.label) for sb in babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
       idx=[nonzero(idx[sb])[0] for sb in range(len(babies))]#.values()]              # get the indices where True
       X_labeled=[val[idx[sb],:] for sb, val in enumerate(FeatureMatrix_for_choosen_patients)]   #selecting the datapoints in label
       y_labeled=[val[idx[sb],:] for sb, val in enumerate(AnnotMatrix_for_choosen_patients)] #get the values for y from idx and label
           
#CALCULATE CLASS WEIGHTS   
       Weights_dict=sample_weight_calc_per_class(y_labeled)
       Var.weight_dict=Weights_dict # saving it in Var
       Weights_all=create_sample_Weigth(y_labeled,Weights_dict,Var.label)
       Var.weight_all=Weights_all
#        weights_all=[[weights_dict.get(x, 0) for x in sublist.flatten()] for sublist in y_labeled]

# ONE HOT ENCODING OF Y-LABELS      
       y_labeled = [np_utils.to_categorical(y_labeled[i],num_classes=Var.label[-1]+1) for i in range(len(y_labeled)) ]      
#       y_labeled = [np_utils.to_categorical(Var.label) for i in range(len(y_labeled)) ]      

#PREPARING SHIFT FOR FOLD       
       Testing_Train_Val_Test=list(percentage_split(X_labeled,Var.split)) 
       vallength=len(Testing_Train_Val_Test[1]); print('possible Val folds: %i' %(floor(len(Testing_Train_Val_Test[0])/vallength) ))       
       if len(Testing_Train_Val_Test)==3:
              testength=len(Testing_Train_Val_Test[2]);print('possible Test folds: %i' %(floor(len(Testing_Train_Val_Test[0])/testength) ))
              
#FOLDING------------------------------------------------------------------------
#-------------------------------------------------------------------------------
       for V in range(Var.fold):
           print('**************************')
           print('Validating on fold: %i' %(V+1) ) 
           Var.Fold=V
# SHIFT FOR EACH FOLD
           if V > 0: 
                  X_labeled=shiftRbyn(X_labeled,vallength)
                  y_labeled=shiftRbyn(y_labeled,vallength)
                  Weights_all=shiftRbyn(Weights_all,vallength)
                  
                  if len(Testing_Train_Val_Test)==3:
                         X_labeled=shiftRbyn(X_labeled,testength)
                         y_labeled=shiftRbyn(y_labeled,testength)
                         Weights_all=shiftRbyn(Weights_all,testength)                         
#ZEROPADDING IF TOTAL SET IS USED AS TIMESTEP           
           if Var.Lookback==1337:
                  list_len = [len(i) for i in X_labeled] # zero pad all sessions/patientsets to have same length
                  X_labeled=[np.pad(X_labeled[i],pad_width=((0,max(list_len)-len(X_labeled[i])),(0,0)),mode ='constant',constant_values=Var.mask_value) for i in range(len(X_labeled))] #padwith=((before axe0, after axe0),(before axe1, before axe1)))
                  y_labeled=[np.pad(y_labeled[i],pad_width=((0,max(list_len)-len(y_labeled[i])),(0,0)),mode ='constant') for i in range(len(y_labeled))] #padwith=((before axe0, after axe0),(before axe1, before axe1)))
                  Weights_all=[np.pad(Weights_all[i],pad_width=((0,max(list_len)-len(Weights_all[i])),(0,0)),mode ='constant') for i in range(len(Weights_all))] #padwith=((before axe0, after axe0),(before axe1, before axe1)))
# SPLIT DATASET                     
           X_Train_Val_Test=list(percentage_split(X_labeled,Var.split)) # Splitting the data into Train-Val-Test after the split percentages
           Y_Train_Val_Test=list(percentage_split(y_labeled,Var.split))
           Weigths_split=list(percentage_split(Weights_all,Var.split))
           
#CREATE LOOCKBACK 3D TENSOR           
           if Var.Lookback!=1337:
                  X_Train=data_with_lookback(X_Train_Val_Test[0],Var.Lookback,666) # function defined above. Split data into loockback steps and create 3D tensor
                  Y_Train=data_with_lookback(Y_Train_Val_Test[0],Var.Lookback,0)
                  Weigths=data_with_lookback(Weigths_split[0],Var.Lookback,0);Weigths=np.squeeze(Weigths, axis=(2,)) # Wheights need to be in shape (samples, sequence_length) for temporal mode
                  
                  X_Val=data_with_lookback(X_Train_Val_Test[1],Var.Lookback,666)
                  Y_Val=data_with_lookback(Y_Train_Val_Test[1],Var.Lookback,0)
                  
                  if len(X_Train_Val_Test)==3:
                         X_Test=data_with_lookback(X_Train_Val_Test[2],Var.Lookback,666)
                         Y_Test=data_with_lookback(Y_Train_Val_Test[2],Var.Lookback,0)
                  else: 
                         X_Test=X_Val
                         Y_Test=Y_Val                         
                  
#CREATE 3D TENSOR FOR TOTAL SET (LOOKBACK= TOTAL SESSION/DATA LENGTH)
           if Var.Lookback==1337: # for any lookbackstep smaler than the total session/set
                  X_Train=X_Train_Val_Test[0]  ;  X_Train = np.stack((X_Train), axis=0)        # here again as we first had to zeropad, then split, then stack       
                  Y_Train=Y_Train_Val_Test[0]  ;  Y_Train = np.stack((Y_Train), axis=0)          
                  Weigths=Weigths_split[0] ; Weigths=np.squeeze(Weigths, axis=(2,)) # Wheights need to be in shape (samples, sequence_length) for temporal mode
               
                  X_Val=X_Train_Val_Test[1]    ;  X_Val   = np.stack((X_Val), axis=0)             
                  Y_Val=Y_Train_Val_Test[1]    ;  Y_Val   = np.stack((Y_Val), axis=0)           
              
                  if len(X_Train_Val_Test)==3:
                         X_Test=X_Train_Val_Test[2]   ;  X_Test  = np.stack((X_Test), axis=0)         
                         Y_Test=Y_Train_Val_Test[2]   ;  Y_Test  = np.stack((Y_Test), axis=0) 
                  else: 
                         X_Test=X_Val
                         Y_Test=Y_Val
               
           Var.class_weights=Weigths  
#CALCULATE CLASS_WEIGHTS TO BALANCE CLASS IMBALANCE               
#           stackedTrain=np.concatenate(Y_Train_Val_Test[0], axis=0)  # make one hot encodd back to integer values representing the labels 
#           indEy = (np.argmax(stackedTrain, axis=1), stackedTrain.shape)
#           Y_train_for_balance=indEy[0]  
#           class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train_for_balance), Y_train_for_balance)
#           
#           class_weights_one_hot=np.zeros((max(label)+1,1))# now put the calulated label weights at the right place/idx for the one hot encoded target
#           for i in range(len(label)):
#                  class_weights_one_hot[label[i]]=class_weights[i] 
#           class_weights_one_hot=dict(enumerate(class_weights_one_hot)) # it seems that Keras likes dicts. I am not 100% sure if that is the latest info or if an array also works
#FORWARD SETS TO KERAS WHERE THE MODEL IS BUILT, TRAINED, VALIDATED AND TESTED           
           print ('Training data shape is:[%i, %i, %i]' %(X_Train.shape))
           model,callbackmetric=KeraS(X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test, Var)

#GATHERING THE RESULTS OF THE TESTING. hre we stack all folds
           if V==0:             
               Ergebnisse.val_f1=callbackmetric.val_f1
               Ergebnisse.val_k=callbackmetric.val_k
               Ergebnisse.val_recall=callbackmetric.val_recall
               Ergebnisse.val_precision=callbackmetric.val_precision
               Ergebnisse.val_no_mask_acc=callbackmetric.val_accuracy_own
               
               Ergebnisse.train_f1=callbackmetric.train_f1
               Ergebnisse.train_k=callbackmetric.train_k
               Ergebnisse.train_recall=callbackmetric.train_recall
               Ergebnisse.train_precision=callbackmetric.train_precision
               Ergebnisse.train_no_mask_acc=callbackmetric.train_accuracy_own
               
               Ergebnisse.train_metric=callbackmetric.train_metric
               Ergebnisse.train_loss=callbackmetric.train_loss
               Ergebnisse.val_metric=callbackmetric.val_metric
               Ergebnisse.val_loss=callbackmetric.val_loss
           if V>0:
               Ergebnisse.val_f1=np.vstack((Ergebnisse.val_f1, callbackmetric.val_f1))
               Ergebnisse.val_k=np.vstack((Ergebnisse.val_k, callbackmetric.val_k))
               Ergebnisse.val_recall=np.vstack((Ergebnisse.val_recall, callbackmetric.val_recall))
               Ergebnisse.val_precision=np.vstack((Ergebnisse.val_precision, callbackmetric.val_precision))
               Ergebnisse.val_no_mask_acc=np.vstack((Ergebnisse.val_no_mask_acc, callbackmetric.val_accuracy_own))
               
               Ergebnisse.train_f1=np.vstack((Ergebnisse.train_f1, callbackmetric.train_f1))
               Ergebnisse.train_k=np.vstack((Ergebnisse.train_k, callbackmetric.train_k))
               Ergebnisse.train_recall=np.vstack((Ergebnisse.train_recall, callbackmetric.train_recall))
               Ergebnisse.train_precision=np.vstack((Ergebnisse.train_precision, callbackmetric.train_precision))
               Ergebnisse.train_no_mask_acc=np.vstack((Ergebnisse.train_no_mask_acc, callbackmetric.train_accuracy_own))
               
               Ergebnisse.train_metric=np.vstack((Ergebnisse.train_metric, callbackmetric.train_metric))
               Ergebnisse.train_loss=np.vstack((Ergebnisse.train_loss, callbackmetric.train_loss))
               Ergebnisse.val_metric=np.vstack((Ergebnisse.val_metric, callbackmetric.val_metric))
               Ergebnisse.val_loss=np.vstack((Ergebnisse.val_loss, callbackmetric.val_loss))
               
#After all folds are collected, create mean               
       if Var.fold>1:
           Ergebnisse.mean_val_metric=np.mean(Ergebnisse.val_metric,axis=0) # Kappa is not calculated per epoch but just per fold. Therefor we generate on mean Kappa
           Ergebnisse.mean_train_metric=np.mean(Ergebnisse.train_metric,axis=0)
           Ergebnisse.mean_val_loss=np.mean(Ergebnisse.val_loss,axis=0)  
           Ergebnisse.mean_train_loss=np.mean(Ergebnisse.train_loss,axis=0)
       
           Ergebnisse.mean_val_f1=np.mean(Ergebnisse.val_f1,axis=0)
           Ergebnisse.mean_val_k=np.mean(Ergebnisse.val_k,axis=0)
           Ergebnisse.mean_val_recall=np.mean(Ergebnisse.val_recall,axis=0)
           Ergebnisse.mean_val_precision=np.mean(Ergebnisse.val_precision,axis=0)
           Ergebnisse.mean_val_no_mask_acc=np.mean(Ergebnisse.val_no_mask_acc,axis=0)
           Ergebnisse.mean_train_f1=np.mean(Ergebnisse.train_f1,axis=0)
           Ergebnisse.mean_train_k=np.mean(Ergebnisse.train_k,axis=0)
           Ergebnisse.mean_train_recall=np.mean(Ergebnisse.train_recall,axis=0)
           Ergebnisse.mean_train_precision=np.mean(Ergebnisse.train_precision,axis=0)
           Ergebnisse.mean_train_no_mask_acc=np.mean(Ergebnisse.train_no_mask_acc,axis=0)
       
#           if plotting:
#                  t_a.append(np.linspace(0,len(y_each_patient_test[V])*30/60,len(y_each_patient_test[V])))
#                  if not compare:
#                         plt.figure(V) 
##                         plt.plot(t_a[V],y_each_patient_test[V])
#                         plt.plot(t_a[V],classpredictions[V]+0.04)
#                         plt.title([V])    
#                  if compare:
#                         plt.figure(V) 
#                         plt.plot(t_a[V],classpredictions[V]+0.07)
#                         plt.title([V])
                   
       """
       ENDING stuff
       """
       
       return model,Ergebnisse            
              
              
  


         