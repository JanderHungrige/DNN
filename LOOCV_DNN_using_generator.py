# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:26:01 2017

@author: 310122653
"""
from DNN_routines import KeraS
from DNN_routines import KeraS_Gen

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

def leave_one_out_cross_validation(\
         babies,AnnotMatrix_each_patient,FeatureMatrix_each_patient,\
         label,classweight, Used_classifier, drawing, lst,ChoosenKind,\
         SamplingMeth,probability_threshold,ASprobLimit, plotting,compare,\
         saving, N,crit,msl,deciding_performance_measure,dispinfo,\
         lookback,split,fold,batchsize,Epochs,dropout,hidden_units):
       
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
           
# ONE HOT ENCODING OF Y-LABELS      
           y_labeled = [np_utils.to_categorical(y_labeled[i]) for i in range(len(y_labeled)) ]           
           
# SPLIT DATASET                     
           X_Train_Val_Test=list(percentage_split(X_labeled,split)) # Splitting the data into Train-Val-Test after the split percentages
           Y_Train_Val_Test=list(percentage_split(y_labeled,split))
                                          
#CALCULATE CLASS_WEIGHTS TO BALANCE CLASS IMBALANCE               
           stackedTrain=np.concatenate(Y_Train_Val_Test[0], axis=0)  # make one hot encodd back to integer values representing the labels 
           indEy = (np.argmax(stackedTrain, axis=1), stackedTrain.shape)
           Y_train_for_balance=indEy[0]  
           class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train_for_balance), Y_train_for_balance)
           
           class_weights_one_hot=np.zeros((max(label)+1,1))# now put the calulated label weights at the right place/idx for the one hot encoded target
           for i in range(len(label)):
                  class_weights_one_hot[label[i]]=class_weights[i] 
           class_weights_one_hot=dict(enumerate(class_weights_one_hot)) # it seems that Keras likes dicts. I am not 100% sure if that is the latest info or if an array also works
           
#FORWARD SETS TO KERAS WHERE THE MODEL IS BUILT, TRAINED, VALIDATED AND TESTED           
           steps_per_epoch = ceil( len(X_Train_Val_Test[0])/ batchsize)
           
           resultsK_fold, mean_k_fold, mean_train_metric_fold, mean_val_metric_fold, mean_train_loss_fold, mean_val_loss_fold, mean_test_metric_fold, mean_test_loss_fold\
           =KeraS_Gen(X_Train_Val_Test, Y_Train_Val_Test,
                      steps_per_epoch, Epochs,dropout,hidden_units,label,class_weights_one_hot,lookback)
           resultsK_fold, mean_k_fold, mean_train_metric_fold, mean_val_metric_fold, mean_train_loss_fold, mean_val_loss_fold, mean_test_metric_fold, mean_test_loss_fold\           
           =KeraS(X_train, Y_train, X_val, Y_val, X_test, Y_test, batchsize,Epochs,dropout,hidden_units,label,class_weight):


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
              
              
  


         