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
         N,crit,msl,deciding_performance_measure,dispinfo,loockback):
       
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
       
       for V in range(len(babies)):
           print('**************************')
           print('Validating on patient: %i' %(V+1) )
           if dispinfo:
                  print('Validating on patient: %i' %(V+1) )

#SPLITTING DATA INTO TRAIN+VALIDATION AND TEST SETS          
           # Using only the Patients choosen in the wrapper
           FeatureMatrix_for_choosen_patients=[FeatureMatrix_each_patient[k] for k in babies]          # get the feature values for selected babies                  
           AnnotMatrix_for_choosen_patients=[AnnotMatrix_each_patient[k] for k in babies]              # get the annotation values for selected babies
           # Using only the labels choosen in the wrapper
           idx=[in1d(AnnotMatrix_each_patient[sb],label) for sb in babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
           idx=[nonzero(idx[sb])[0] for sb in range(len(babies))]#.values()]              # get the indices where True
           X_labeled=[val[idx[sb],:] for sb, val in enumerate(FeatureMatrix_for_choosen_patients)]   #selecting the datapoints in label
           y_labeled=[val[idx[sb],:] for sb, val in enumerate(AnnotMatrix_for_choosen_patients)] #get the values for y from idx and label    
                          
           # Selecting what becomes testing and what train and validation
           selected_babies_train=list(delete(babies,babies[V]))# Babies to train and test on ; j-1 as index starts with 0
           selected_babies_test=babies[V]# Babies to validate on 
           
           X_train_val=[X_labeled[k] for k in selected_babies_train]          # get the feature values for selected babies           
           y_train_val=[y_labeled[k] for k in selected_babies_train]              # get the annotation values for selected babies
           
           X_test=X_labeled[selected_babies_test]          # get the feature values for selected babies
           y_test=y_labeled[selected_babies_test]               # get the annotation values for selected babies
           
#CREATING TENSOR FOR INPUT INCOPORATING LOOCKBACK FOR LSTM                    
        

           X_train_val, y_train_val=create_Tensor_with_lookback(X_train_val,y_train_val,loockback)   
           X_test, y_test=create_Tensor_with_lookback(X_test,y_test,loockback)   
   
           fold=3
#FORWARD SETS TO KERAS WHERE THE MODEL IS BUILT, TRAINED, VALIDATED AND TESTED           
           #Validate with left out patient 
           # Run the classifier with the selected FEature subset in selecteF
           resultsK_fold, mean_k_fold, mean_train_metric_fold, mean_val_metric_fold, mean_train_loss_fold, mean_val_loss_fold, mean_test_metric_fold, mean_test_loss_fold\
           =KeraS(X_train_val, y_train_val,X_test, y_test, selected_babies_train, \
                               label, fold)

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
              
              
  


         