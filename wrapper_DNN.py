# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:37:59 2018

@author: 310122653

The data is already standard scaled (z-scale mean=1 std=0) in MAtlab. For each session separatelly. Nan is removed from the Matrices.
11.5.2018


"""

from platform import python_version
print ('Python version: ', sep=' ', end='', flush=True);print( python_version())	


#from Loading_5min_mat_files_cECG import \
#babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient_all, Class_dict, features_dict, features_indx, \
#FeatureMatrix_each_patient_fromSession, lst
#from Classifier_routines import Classifier_random_forest
from Loading_5min_mat_files_DNN import Loading_data_all,Loading_data_perSession,Feature_names,Loading_Annotations
from LOOCV_DNN import leave_one_out_cross_validation
#from LOOCV_DNN_using_generator import leave_one_out_cross_validation

import itertools
from matplotlib import *
#from numpy import *
from pylab import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score 
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn import svm, cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Perceptron
import sys #to add strings together
import pdb # use pdb.set_trace() as breakpoint



#from compute_class_weight import *   
import time
start_time = time.time()
Klassifier=['RF','ERF','TR','GB', 'LR']
SampMeth=['NONE','SMOTE','ADASYN']
Whichmix=['perSession', 'all']


#_Labels_ECG_Featurelist_Scoring_classweigt_C_gamma

description='_123456_cECG_lst_micro_'
consoleinuse='4'
dispinfo=0
"""
**************************************************************************
Loading data declaration & Wrapper variables
0,1,2 = ECG
3,4,5,6,7,8,9,10,11,12,13,14,15,16,17= HRV time domain
18,19,20,21,22,23,24,25,26,27,28 = HRV freq domain
29,30,31,32,33= HRV nonlinear
**************************************************************************
"""
class Variablen:
       FeatureSet='Features' #Features ECG, EDR, HRV
       lst= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33] 
       label=[1,2,3,4,6] # 1=AS 2=QS 3=Wake 4=Care-taking 5=NA 6= transition
       usedPC='Philips' #Philips or c3po or Cluster
       dataset='cECGDNN'  #"ECG" "cECG" "MMC" "MMC+cECG" "cECGDNN"
       merge34=1
       WhichMix='all' #perSession or all  # determine how the data was scaled. PEr session or just per patient
       saving_model=1
       
       
       Lookback= 1337# 1337 or anything else . #Loockback for the LSTM. The data is separated samples with timestep=loockback; #Loockback of 1337 mean all data per patient. Otherwise it is in nr of 30s epochs. e.g. 60=30min  120=1h 10=5min
#       split=[0.60,0.2,0.2];# HOw to split the dataset in [Train, Validation, Test] e.g.70:15:15  or 50:25:25 ,... # The split is done for each fold. Just for the chekout phase use fold one. Later calculate how often the test split fits into the total data, that is the fold. e.g. 30 patients with 15% test -> 4.5 (round to 5) patients per fold. Now see how many times the 30 can be folded with 5 patients in the test set to cover all patients. 30/5=6 -> 6 fold
       split=[0.70,0.30];
       batchsize=5  # LSTM needs [batchsize, timestep, feature] your batch size divides nb_samples from the original tensor. So batchsize should be smaller than samples
       Epochs=200
       hidden_units=32 # 2-64 or even 1000 as used by sleepnet best: multible of 32
       Dense_Unit=34
       dropout=0.5 #0.5; 0.9  dropout can be between 0-1  as %  DROPOUT CAN BE ADDED TO EACH LAYER
       learning_rate=0.001 #0.0001 to 0.01 default =0.001
       learning_rate_decay=0.0 #0.0 default
       fold=2
       scalerange=(0, 2) #(0,1) or (-1,1) #If you are using sigmoid activation functions, rescale your data to values between 0-and-1. If you’re using the Hyperbolic Tangent (tanh), rescale to values between -1 and 1.
       scaler = MinMaxScaler(feature_range=scalerange) #define function
       Loss_Function='categorical_crossentropy'# categorical_crossentropy OR mean_squared_error IF BINARY : binary_crossentropy
       Perf_Metric=['categorical_accuracy']# 'categorical_accuracy' OR 'binary_accuray'
       activationF='sigmoid' # 'relu', 'tanh', 'sigmoid' ,...  Only in case the data is not normalized , only standardised
       Kr=0.01 # Kernel regularizers
       Ar=0.01 #ACtivity regularizers
       residual_blocks=1
        
Var=Variablen()    


if Var.dataset=='ECG' or 'cECG' or 'cECGDNN':
         Var.selectedbabies =[0,1,2,3,5,6,7,8] #0-8 ('4','5','6','7','9','10','11','12','13')
if Var.dataset == 'MMC':
       Var.selectedbabies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] #0-21
if Var.dataset == 'MMC+cECG':
       Var.selectedbabies=[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] #0-27    # first 10 cECG rest MMC   

if Var.scalerange==(0,1) :
       Var.activationF='sigmoid'
elif Var.scalerange==(-1,1):
       Var.activationF='tanh'    

if len(Var.label)==2:
       Var.Loss_Function='binary_crossentropy'       
       Var.Perf_Metric=['categorical_accuracy'] # ['categorical_accuracy'] OR ['binary_accuray']
 
if Var.Lookback==1337: # The problem is that the patients have different lenght. Then we need to zero pad. Instead of zeropadding we can use diffent length when batchsize==1
       Var.batchsize=1
       
if Var.merge34 and 3 in Var.label:
              Var.label.remove(3)      
info={'label': Var.label,'Features':'all','Lookback': Var.Lookback,'split': Var.split,'batchsize': Var.batchsize,'Epochs': Var.Epochs,'hidden_units':Var.hidden_units, 
      'dropout': Var.dropout,'learning_rate': Var.learning_rate,'learning_rate_decay': Var.learning_rate_decay, 
      'fold': Var.fold, 'Scale': Var.scalerange,'Loss_Function': Var.Loss_Function,
      'Perf_Metric': Var.Perf_Metric,'Activation_funtion': Var.activationF,'Dens_unit': Var.Dense_Unit, 
      'Kernel Regularizer': Var.Kr , 'Activity regularizer': Var.Ar}
#---------------------------
#AVERAGING
class Variablenplus:
       FensterQS=20 
       ExFeatQS=1; 
       FEATaQS=[Var.lst.index(0),Var.lst.index(25),Var.lst.index(33)]# 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26

#POLY
       PolyTQS=0; 
       FEATpQS=[11,14,29,30,31]#[lstIS.index(11),lstIS.index(12),lstIS.index(13),lstIS.index(14),lstIS.index(10),lstIS.index(24),lstIS.index(29),lstIS.index(32)]#[0,3,4,5]#12

       ASQS= [0,0.69]#[0.65,0]#[0.63,0.7]
       NQS=100; mslQS=5 #100 2

       Rpeakmethod='R' #R or M

       Movingwindow=FensterQS # WIndow size for moving average
       preaveraging=0
       postaveraging=0
       exceptNOF=ExFeatQS #Which Number of Features (NOF) should be used with moving average?  all =oth tzero; only some or all except some defined in FEAT
       onlyNOF=0 # [0,1,2,27,28,29]
       FEAT=FEATaQS
#----------------------------
       PolyTrans=PolyTQS#use polinominal transformation on the Features specified in FEATp
       ExpFactor=2# which degree of polinomonal (2)
       exceptNOpF= 0#Which Number of Features (NOpF) should be used with polynominal fit?  all =0; only some or all except some defined in FEATp
       onlyNOpF=1 # [0,1,2,27,28,29]
       FEATp=FEATpQS
#---------------------------
       SamplingMeth='NONE'  # 'NONE' 'SMOTE'  or 'ADASYN' #For up and downsampling of data
       ChoosenKind=0   # 0-3['regular','borderline1','borderline2','svm'] only when using SMOTE
#---------------------------
       probability_threshold=1 # 1 to use different probabilities tan 0.5 to decide on the class. At the moment it is >=0.2 for any other calss then AS
       ASprobLimit=ASQS# Determine the AS lower limit for the probability for which another class is chosen than AS. For: [3 labels, >3 labels]


Varplus=Variablenplus()
# fix random seed for reproducibility
np.random.seed(7)
#%%
"""
Loading Data
"""
Class_dict, features_dict, features_indx=Feature_names()

# CHOOSING WHICH FEATURE MATRIX IS USED
def loading_and_DNN(whichMix):
       if Var.WhichMix=='perSession':            
              babies, AnnotMatrix_each_patient,FeatureMatrix_each_patient\
              =Loading_data_perSession(Var,Varplus)       
              
       elif Var.WhichMix=='all':              
              babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient\
              =Loading_data_all(Var,Varplus)                   

       return  babies, AnnotMatrix_each_patient,FeatureMatrix_each_patient

babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient\
= loading_and_DNN(Var)  

#%%
"""
LOOCV 
"""    
laenge=[sum(len(FeatureMatrix_each_patient[i]) for i in range(len(FeatureMatrix_each_patient))) ]
print('Total amount of epochs: {}'.format(laenge))

model,y_each_patient, mean_Kappa, mean_train_metric, mean_train_loss, mean_val_metric, mean_val_loss, mean_test_metric, mean_test_loss\
=leave_one_out_cross_validation(babies,AnnotMatrix_each_patient,FeatureMatrix_each_patient,Var,Varplus)

if Var.fold>1:
       mean_test_metric_overall=np.mean(mean_test_metric) # Kappa is not calculated per epoch but just per fold. Therefor we generate on mean Kappa
       mean_train_metric_overall=np.mean(mean_train_metric,axis=0)
       mean_val_metric_overall=np.mean(mean_val_metric,axis=0)    
       mean_test_loss_overall=mean(mean_test_loss,axis=0)    
       mean_train_loss_overall=np.mean(mean_train_loss,axis=0)
       mean_val_loss_overall=np.mean(mean_val_loss,axis=0)      
       mean_Kappa_overall=np.mean(mean_Kappa)

if Var.saving_model:
       from keras.models import load_model
       model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#

## Kappa over all annotations and predictions merged together
#tmp_orig=vstack(y_each_patient)
#tmp_pred=hstack(classpredictions)
#
##Performance of optimized predictions 
#RES_MEA_all=zeros(shape=(len(babies),len(label)))
#KonfMAT=list()
#KonfMATall=list()
#RES_Kappa=list()
#
#for K in range(len(babies)):
#       KonfMAT.append(confusion_matrix(y_each_patient[K].ravel(), classpredictions[K], labels=label, sample_weight=None))
#       RES_Kappa.append(cohen_kappa_score(y_each_patient[K].ravel(),classpredictions[K],labels=label)) # Find the threshold where Kapaa gets max
#RES1_kappa_STD=std(RES1_Kappa)       
#RES1_Kappa.append(mean(RES1_Kappa))
#RES1_F1_all_mean=array(mean(RES1_F1_all,0))    
#RES1_KAPPA_overall=cohen_kappa_score(tmp_orig.ravel(),tmp_pred.ravel(),labels=label)
#KonfMATall.append(confusion_matrix(tmp_orig.ravel(), tmp_pred.ravel(), labels=label, sample_weight=None))
#

#PlottingFeatureImportance(Fimportance_QS,Fimportance_CT,Fimportance_IS,features_dict)

"""
END
"""
disp('-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - ')

if len(Var.split)<3:
       disp('don`t give a f** about Kappa. Data is only plit for train and val' )
import time
t=time.localtime()
zeit=time.asctime()
Minuten=(time.time() - start_time)/60
Stunden=(time.time() - start_time)/3600
print('FINISHED Console ' + consoleinuse)
print("--- %i seconds ---" % (time.time() - start_time))
print("--- %i min ---" % Minuten)
print("--- %i h ---" % Stunden)


#disp(  RES1_Kappa[-1])
#disp(RES1_kappa_STD)
#disp (RES1_KAPPA_overall)
