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

def leave_one_out_cross_validation(babies,AnnotMatrix_each_patient,FeatureMatrix_each_patient,\
         label,classweight, Used_classifier, drawing, lst,ChoosenKind,SamplingMeth,probability_threshold,ASprobLimit,\
         plotting,compare,saving,\
         N,crit,msl,deciding_performance_measure,dispinfo,lookback,split,fold, batchsize):
       
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
                     
     
       def data_with_lookback(Data_Train_Val_Test,lookback):
              Data_tmp=concatenate(Data_Train_Val_Test,axis=0)
              Data_Tensor=list(create_timesteps(Data_tmp, lookback))
              if Data_Tensor[-1].shape < Data_Tensor[-2].shape:  #make the last one the same length as the others, zeropad                       
                         Data_Tensor[-1]=np.pad(Data_Tensor[-1],pad_width=((0,Data_Tensor[-2].shape[0]-Data_Tensor[-1].shape[0]),(0,0)),mode='constant')
              Data_Tensor = np.stack((Data_Tensor), axis=0) 
              return Data_Tensor                      
#               
#      def percentage_split(seq, percentages):
#          cdf = cumsum(percentages)
#          assert cdf[-1] == 1.0
#          stops = map(int, cdf * len(seq))
#          return [seq[a:b] for a, b in zip([0]+stops, stops)]               
        
       
       for V in range(fold):
           print('**************************')
           print('Validating on fold: %i' %(V+1) )


#SPLITTING DATA INTO TRAIN+VALIDATION AND TEST SETS
           # Using only the Patients choosen in the wrapper
           FeatureMatrix_for_choosen_patients=[FeatureMatrix_each_patient[k]for k in babies]          # get the feature values for selected babies                  
           AnnotMatrix_for_choosen_patients=[AnnotMatrix_each_patient[k] for k in babies]              # get the annotation values for selected babies
           

           # Using only the labels choosen in the wrapper
           idx=[in1d(AnnotMatrix_each_patient[sb],label) for sb in babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
           idx=[nonzero(idx[sb])[0] for sb in range(len(babies))]#.values()]              # get the indices where True
           X_labeled=[val[idx[sb],:] for sb, val in enumerate(FeatureMatrix_for_choosen_patients)]   #selecting the datapoints in label
           y_labeled=[val[idx[sb],:] for sb, val in enumerate(AnnotMatrix_for_choosen_patients)] #get the values for y from idx and label    
#ZEROPADDING IF TOTAL SET IS USED AS TIMESTEP           
           if lookback==1337:
                  list_len = [len(i) for i in X_labeled] # zero pad all sessions/patientsets to have same length
                  X_labeled=[np.pad(X_labeled[i],pad_width=((0,max(list_len)-len(X_labeled[i])),(0,0)),mode ='constant',constant_values=666) for i in range(len(X_labeled))] #padwith=((before axe0, after axe0),(before axe1, before axe1)))
                  y_labeled=[np.pad(y_labeled[i],pad_width=((0,max(list_len)-len(y_labeled[i])),(0,0)),mode ='constant') for i in range(len(y_labeled))] #padwith=((before axe0, after axe0),(before axe1, before axe1)))
                  
# SPLIT DATASET     
                  
           X_Train_Val_Test=list(percentage_split(X_labeled,split)) # Splitting the data into Train-Val-Test after the split percentages
           Y_Train_Val_Test=list(percentage_split(y_labeled,split))
#CREATE LOOCKBACK 3D TENSOR           
           if lookback!=1337:
                  X_Train=data_with_lookback(X_Train_Val_Test[0],lookback) # function defined above. Split data into loockback steps and create 3D tensor
                  Y_Train=data_with_lookback(Y_Train_Val_Test[0],lookback)
                  
                  X_Val=data_with_lookback(X_Train_Val_Test[1],lookback)
                  Y_Val=data_with_lookback(Y_Train_Val_Test[1],lookback)

                  X_Test=data_with_lookback(X_Train_Val_Test[2],lookback)
                  Y_Test=data_with_lookback(Y_Train_Val_Test[2],lookback)
                  
#CREATE 3D TENSOR FOR TOTAL SET (LOOKBACK= TOTAL SESSION/DATA LENGTH)
           if lookback==1337: # for any lookbackstep smaler than the total session/set
               X_train=X_Train_Val_Test[0]  ;  X_train = np.stack((X_train), axis=0)              
               Y_train=Y_Train_Val_Test[0]  ;  Y_train = np.stack((Y_train), axis=0)          
           
               X_val=X_Train_Val_Test[1]    ;  X_val   = np.stack((X_val), axis=0)             
               Y_val=Y_Train_Val_Test[1]    ;  Y_val   = np.stack((Y_val), axis=0)           
           
               X_test=X_Train_Val_Test[2]   ;  X_test  = np.stack((X_test), axis=0)         
               Y_test=Y_Train_Val_Test[2]   ;  Y_test  = np.stack((Y_test), axis=0)           
           

#FORWARD SETS TO KERAS WHERE THE MODEL IS BUILT, TRAINED, VALIDATED AND TESTED           

           resultsK_fold, mean_k_fold, mean_train_metric_fold, mean_val_metric_fold, mean_train_loss_fold, mean_val_loss_fold, mean_test_metric_fold, mean_test_loss_fold\
           =KeraS(X_train, Y_train, X_val, Y_val, X_test, Y_test, batchsize,label)

#GATHERING THE RESULTS OF THE TESTING           
#           classpredictions.append(prediction)
           Performance_K.append(mean_k_fold)
           mean_train_metric.append(mean_train_metric_fold)
           mean_train_loss.append(mean_train_loss_fold)
           mean_val_metric.append(mean_val_metric_fold)
           mean_val_loss.append(mean_val_loss_fold)
           mean_test_metric.append(mean_test_metric_fold)
           mean_test_loss.append(mean_test_loss_fold)
       
           if plotting:
                  t_a.append(np.linspace(0,len(y_each_patient_test[V])*30/60,len(y_each_patient_test[V])))
                  if not compare:
                         plt.figure(V) 
#                         plt.plot(t_a[V],y_each_patient_test[V])
                         plt.plot(t_a[V],classpredictions[V]+0.04)
                         plt.title([V])    
                  if compare:
                         plt.figure(V) 
                         plt.plot(t_a[V],classpredictions[V]+0.07)
                         plt.title([V])
                   
       """
       ENDING stuff
       """

       return y_labeled,\
              Performance_K,\
              mean_train_metric,\
              mean_train_loss,\
              mean_val_metric,\
              mean_val_loss,\
              mean_test_metric,\
              mean_test_loss   
              
              
  


         