# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:26:01 2017

@author: 310122653
"""
from Classifier_routines import kEras

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
         N,crit,msl,deciding_performance_measure,dispinfo):
       
       t_a=list()
#       classpredictions=list()
#        
#       for F in babies:
#              classpredictions=list((zeros(shape=(len(babies),len(FeatureMatrix_each_patient[F])))))
       classpredictions=list()
       Probabilities=list()
       Performance=list()
       ValidatedPerformance_macro=list()
       ValidatedPerformance_K=list()
       ValidatedPerformance_micro=list()
       ValidatedPerformance_weigth=list()  
       ValidatedPerformance_all=zeros(shape=(len(babies),len(label)))
       ValidatedFimportance=zeros(shape=(len(babies),len(FeatureMatrix_each_patient[0][1])))
       Validatedscoring=list()        
       
       for V in range(len(babies)):
           print('**************************')
           if dispinfo:
                  print('Validating on patient: %i' %(V+1) ) 
           
           selected_babies=list(delete(babies,babies[V]))# Babies to train and test on ; j-1 as index starts with 0
           selected_test=babies[V]# Babies to validate on 
           #### Create Matrices for selected babies
           
           AnnotMatrix_auswahl=[AnnotMatrix_each_patient[k] for k in selected_babies]              # get the annotation values for selected babies
           FeatureMatrix_auswahl=[FeatureMatrix_each_patient[k] for k in selected_babies]          # get the feature values for selected babies
           idx=[in1d(AnnotMatrix_each_patient[sb],label) for sb in selected_babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
           idx=[nonzero(idx[sb])[0] for sb in range(len(selected_babies))]#.values()]              # get the indices where True
           Xfeat=[val[idx[sb],:] for sb, val in enumerate(FeatureMatrix_auswahl)]   #selecting the datapoints in label
           y_each_patient=[val[idx[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl) if sb in range(len(selected_babies))] #get the values for y from idx and label    

        
           #creating another Set where all PAtients are included. In the Random Forest function one is selected for training. otherwise dimension missmatch   
           AnnotMatrix_auswahl_test=[AnnotMatrix_each_patient[k] for k in babies]              # get the annotation values for selected babies
           FeatureMatrix_auswahl_test=[FeatureMatrix_each_patient[k] for k in babies]
           idx_test=[in1d(AnnotMatrix_each_patient[sb],label) for sb in babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
           idx_test=[nonzero(idx_test[sb])[0] for sb in range(len(babies))]#.values()]              # get the indices where True
           Xfeat_test=[val[idx_test[sb],:] for sb, val in enumerate(FeatureMatrix_auswahl_test)]  
           y_each_patient_test=[val[idx_test[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl_test) if sb in range(len(babies))] #get the values for y from idx and label
            
       
           #Validate with left out patient 
           # Run the classifier with the selected FEature subset in selecteF
           resultsK,resultsF1_all,prediction,loss_and_metrics \
           =kEras(Xfeat_test, Xfeat,y_each_patient_test, y_each_patient, selected_babies, \
                              selected_test, label, classweight, Used_classifier, lst, ChoosenKind,\
                              SamplingMeth)
       
       #    =Classifier_random_forest(Xfeat,y_each_patient,selected_babies,label,classweight)       
       #    sys.exit('Jan werth 222')
#           classpredictions[V]=prediction
           
           classpredictions.append(prediction)
           ValidatedPerformance_K.append(resultsK)
           ValidatedPerformance_all[V]=resultsF1_all
           Validatedscoring.append(loss_and_metrics)
       
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
       
       ValidatedPerformance_K.append(mean(ValidatedPerformance_K)) 
       ValidatedPerformance_all_mean=array(mean(ValidatedPerformance_all,0))
       
       
           
       return y_each_patient_test,\
              classpredictions,\
              ValidatedPerformance_K,\
              ValidatedPerformance_all,\
              ValidatedPerformance_all_mean,\
              Validatedscoring
              
  


         