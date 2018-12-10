# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:37:59 2018

@author: 310122653

The data is already standard scaled (z-scale mean=1 std=0) in MAtlab. For each session separatelly. Nan is removed from the Matrices.
11.5.2018


"""
#! /bin/sh -p
#PBS -l nodes=1:ppn=1
#PBS -N JobName

from platform import python_version
print ('Python version: ', sep=' ', end='', flush=True);print( python_version())	

#from Loading_5min_mat_files_cECG import \
#babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient_all, Class_dict, features_dict, features_indx, \
#FeatureMatrix_each_patient_fromSession, lst
#from Classifier_routines import Classifier_random_forest
from Loading_5min_mat_files_DNN import Loading_data_all,Loading_data_perSession,Feature_names,Loading_Annotations
from LOOCV_DNN import leave_one_out_cross_validation
from InputValues import inputcombinations
from InputValues2 import inputcombinations2

from DNN_routines import KeraS #import the file as the result clss is defined there which is needed for pickle
 
from send_mail import noticeEMail
import datetime as dt
starttime=dt.datetime.now()
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

import pickle # to save objects

#from compute_class_weight import *   
#import time
start_time = time.time()
Klassifier=['RF','ERF','TR','GB', 'LR']
SampMeth=['NONE','SMOTE','ADASYN']
Whichmix=['perSession', 'all']



#_Labels_ECG_Featurelist_Scoring_classweigt_C_gamma


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
arrayId=int(sys.argv[1])
print('RunningNumber: ' +str(arrayId))


if arrayId in arange(39,207) or arrayId in arange(1,15):
    [descriptionID, datasetID, modelID, labelID, Loss_FunctionID, OptimizerID]=inputcombinations(arrayId)
if arrayId in arange(500,503) or arrayId in arange(300,307) or arrayId in arange(340,442) or arrayId in arange(600,607) or arrayId in arange(700,742)or arrayId in arange(800,836) or arrayId in arange(900,907):
    [descriptionID, datasetID, modelID, labelID, Loss_FunctionID, OptimizerID]=inputcombinations2(arrayId)
if arrayId in arange(1300,1307) or arrayId in arange(2300,2307) or arrayId in arange(1600,1607) or arrayId in arange(2600,2607)or arrayId in arange(1800,1836)or arrayId in arange(2800,2836) or arrayId in arange(1900,1907)or arrayId in arange(2900,2907):
    [descriptionID, datasetID, modelID, labelID, Loss_FunctionID, OptimizerID]=inputcombinations2(arrayId)

SavingResults=1
class Variablen:
       hidden_units=32 # 2-64 or even 1000 as used by sleepnet best: multible of 32
       Dense_Unit=32
       
       runningNumber=str(arrayId) #'125'
       label=labelID #[1,2] # 1=AS 2=QS 3=Wake 4=Care-taking 5=NA 6= transition
       dataset=datasetID  #'MMC+ECG+InSe' #MMC+ECG+InSe     MMC+ECG   MMC+InSe   ECG+InSe   MMC   ECG   InSe         "cECG"   
       model= modelID  #'model_3_LSTM_advanced' # check DNN_routines KeraS for options model_4_GRU_advanced  model_3_LSTM_advanced
       Loss_Function=Loss_FunctionID  #'categorical_crossentropy'#Weighted_cat_crossentropy or categorical_crossentropy OR mean_squared_error IF BINARY : binary_crossentropy
       optimizer=OptimizerID
       usedPC='Cluster' #Philips or c3po or Cluster
       Epochs=1000
       fold=3   

       PatSet='all'  #'023578'  'all'
       mask_value=0.666 
       saving_model=0
       if arrayId in arange(39,207):
           saving_model=1
       SavingResults=1
       if usedPC=='Cluster':
#           resultpath='/home/310122653/Git/DNN/Results/'
           resultpath='/home/310122653/Git/DNN/Results/'
       else:
           resultpath='./Results/'
    
       FeatureSet='Features' #Features ECG, EDR, HRV
       lst= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46] 
       merge34=1
       WhichMix='all' #perSession or all  # determine how the data was scaled. PEr session or just per patient
       early_stopping_patience=100 #epochs
       Jmethod='weighted' # the method used for f1 precicion /recall ...       
       Lookback= 1337# 1337 or anything else . #Loockback for the LSTM. The data is separated samples with timestep=loockback; #Loockback of 1337 mean all data per patient. Otherwise it is in nr of 30s epochs. e.g. 60=30min  120=1h 10=5min
#       split=[0.60,0.2,0.2];# HOw to split the dataset in [Train, Validation, Test] e.g.70:15:15  or 50:25:25 ,... # The split is done for each fold. Just for the chekout phase use fold one. Later calculate how often the test split fits into the total data, that is the fold. e.g. 30 patients with 15% test -> 4.5 (round to 5) patients per fold. Now see how many times the 30 can be folded with 5 patients in the test set to cover all patients. 30/5=6 -> 6 fold
       split=[0.70,0.30];
       batchsize=5  # LSTM needs [batchsize, timestep, feature] your batch size divides nb_samples from the original tensor. So batchsize should be smaller than samples

       dropout=0.5 #0.5; 0.9  dropout can be between 0-1  as %  DROPOUT CAN BE ADDED TO EACH LAYER
       learning_rate=0.001 #0.0001 to 0.01 default =0.001
       learning_rate_decay=0.0 #0.0 default
       scalerange=(0, 2) #(0,1) or (-1,1) #If you are using sigmoid activation functions, rescale your data to values between 0-and-1. If you’re using the Hyperbolic Tangent (tanh), rescale to values between -1 and 1.
       scaler = MinMaxScaler(feature_range=scalerange) #define function
       Perf_Metric=['categorical_accuracy']# 'categorical_accuracy' OR 'binary_accuray'
       activationF='sigmoid' # 'relu', 'tanh', 'sigmoid' ,...  Only in case the data is not normalized , only standardised
       Kr=0.001 # Kernel regularizers
       Ar=0.0001 #ACtivity regularizers
       residual_blocks=1

       
       info={'label': label,'Features':'all','Lookback': Lookback,'split': split,
      'batchsize': batchsize,'Epochs': Epochs,
      'hidden_units':hidden_units, 'Dens_unit': Dense_Unit,
      'dropout': dropout, 'Kernel Regularizer': Kr , 'Activity regularizer': Ar,
      'learning_rate': learning_rate,'learning_rate_decay': learning_rate_decay, 
      'fold': fold, 'Scale': scalerange,'Loss_Function': Loss_Function,
      'Perf_Metric': Perf_Metric,'Activation_funtion': activationF,
      'ResidualBlocks':residual_blocks,'model' :model, 'earlystoppingpatience': early_stopping_patience,
      'Metric weighting method':Jmethod}       
       
       if Loss_FunctionID=='categorical_crossentropy' :
           wID=0
       if Loss_FunctionID=='Weighted_cat_crossentropy1' :
           wID=1     
       if Loss_FunctionID=='Weighted_cat_crossentropy2' :
           wID=2         
           
       if 'LSTM' in model:
           modelNr='3'
       if 'GRU' in model:
           modelNr='4'
#       description= descriptionID+'_2_DU'+str(Dense_Unit)+'_W'+str(wID)+'_'+str(fold)+'_folds_Pat_' + PatSet+'_Mask_'+str(mask_value)#'Bi_ASCTW' # Bi_ASQS  Bi_ASIS  Bi_ASCTW  Bi_QSIS  Bi_QSCTW  Bi_ISCTW
       description= descriptionID+'_Kr_'+str(Kr)+'_Ar_'+str(Ar)+'_drop_'+str(dropout)+'_wID_'+str(wID)+'_Fold_'+str(fold)+'_Model_'+str(modelNr)# Bi_ASQS  Bi_ASIS  Bi_ASCTW  Bi_QSIS  Bi_QSCTW  Bi_ISCTW
      
#%% Var finished --------------------------------------------------------------

       
Var=Variablen()    
print('Dense_Unit: ' + str(Var.Dense_Unit))
print('Hidden_units: ' + str(Var.hidden_units))



if Var.PatSet=='all':
    if Var.dataset=='ECG' or 'cECG' or 'cECGDNN':
#           Var.selectedbabies =[0,1,2,3,5,6,7,8] #0-8 ('4','5','6','7','9','10','11','12','13')
           Var.selectedbabies =[0,1,3,5,6,7,8] #0-8 ('4','5','6','7','9','10','11','12','13')

    if Var.dataset == 'InSe':
           Var.selectedbabies =[0,1,2,3,5,7] #0-7      '3','4','5','6','8','9','13','15'              
    if Var.dataset == 'MMC':
           Var.selectedbabies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] #0-21
    if Var.dataset == 'MMC+ECG':
           Var.selectedbabies=[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] #0-27    # first 9 cECG rest MMC    
    if Var.dataset == 'MMC+InSe':
           Var.selectedbabies=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] #0-27    # first 8 InSen rest MMC    
    if Var.dataset == 'ECG+InSe':
           Var.selectedbabies=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16] #0-17    # first 8 InSe rest cECG               
    if Var.dataset == 'MMC+ECG+InSe':
           Var.selectedbabies= [0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38] #0-17    # first 8 InSe,8-16 cECG, rest MMC           

if Var.PatSet=='01235678':
    if Var.dataset=='ECG' or 'cECG' or 'cECGDNN':
           Var.selectedbabies =[0,1,2,3,5,6,7,8] #0-8 ('4','5','6','7','9','10','11','12','13')
    if Var.dataset == 'InSe':
           Var.selectedbabies =[0,1,2,3,5,7] #0-7      '3','4','5','6','8','9','13','15'              
    if Var.dataset == 'MMC':
           Var.selectedbabies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] #0-21
    if Var.dataset == 'MMC+ECG':
           Var.selectedbabies=[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] #0-27    # first 9 cECG rest MMC    
    if Var.dataset == 'MMC+InSe':
           Var.selectedbabies=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] #0-27    # first 8 InSen rest MMC    
    if Var.dataset == 'ECG+InSe':
           Var.selectedbabies=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16] #0-17    # first 8 InSe rest cECG               
    if Var.dataset == 'MMC+ECG+InSe':
           Var.selectedbabies= [0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38] #0-17    # first 8 InSe,8-16 cECG, rest MMC           
    
if Var.PatSet=='02357':
    if Var.dataset=='ECG' or 'cECG' or 'cECGDNN':
           Var.selectedbabies =[0,2,3,5,7]#[0,1,2,3,5,6,7,8] #0-8 ('4','5','6','7','9','10','11','12','13')
    if Var.dataset == 'InSe':
           Var.selectedbabies =[0,1,2,3,5,7] #0-7      '3','4','5','6','8','9','13','15'              
    if Var.dataset == 'MMC':
           Var.selectedbabies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] #0-21
    if Var.dataset == 'MMC+ECG':
           Var.selectedbabies=[0,2,3,5,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] #[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] #0-27    # first 9 cECG rest MMC    
    if Var.dataset == 'MMC+InSe':
           Var.selectedbabies=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] #0-27    # first 8 InSen rest MMC    
    if Var.dataset == 'ECG+InSe':
           Var.selectedbabies=[0,1,2,3,5,7,8,10,11,13,15]#[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16] #0-17    # first 8 InSe rest cECG               
    if Var.dataset == 'MMC+ECG+InSe':
           Var.selectedbabies=[0,1,2,3,5,7,8,10,11,13,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38] #0-17    # first 8 InSe,8-16 cECG, rest MMC           #[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38] #0-17    # first 8 InSe,8-16 cECG, rest MMC           

if Var.PatSet=='023578':
    if Var.dataset=='ECG' or 'cECG' or 'cECGDNN':
           Var.selectedbabies =[0,2,3,5,7,8]#[0,1,2,3,5,6,7,8] #0-8 ('4','5','6','7','9','10','11','12','13')
    if Var.dataset == 'InSe':
           Var.selectedbabies =[0,1,2,3,5,7] #0-7      '3','4','5','6','8','9','13','15'              
    if Var.dataset == 'MMC':
           Var.selectedbabies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] #0-21
    if Var.dataset == 'MMC+ECG':
           Var.selectedbabies=[0,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] #[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] #0-27    # first 9 cECG rest MMC    
    if Var.dataset == 'MMC+InSe':
           Var.selectedbabies=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] #0-27    # first 8 InSen rest MMC    
    if Var.dataset == 'ECG+InSe':
           Var.selectedbabies=[0,1,2,3,5,7,8,10,11,13,15,16]#[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16] #0-17    # first 8 InSe rest cECG               
    if Var.dataset == 'MMC+ECG+InSe':
           Var.selectedbabies=[0,1,2,3,5,7,8,10,11,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38] #0-17    # first 8 InSe,8-16 cECG, rest MMC           #[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38] #0-17    # first 8 InSe,8-16 cECG, rest MMC           

if Var.scalerange==(0,1) :
       Var.activationF='sigmoid'
elif Var.scalerange==(-1,1):
       Var.activationF='tanh'    

#if len(Var.label)==2:
#       Var.Loss_Function='binary_crossentropy'       
#       Var.Perf_Metric=['categorical_accuracy'] # ['categorical_accuracy'] OR ['binary_accuray']
#       Var.Jmethod='binary'
if Var.Lookback==1337: # The problem is that the patients have different lenght. Then we need to zero pad. Instead of zeropadding we can use diffent length when batchsize==1
       Var.batchsize=1
       
if Var.merge34 and 3 in Var.label:
              Var.label.remove(3)      
              
#info={'label': Var.label,'Features':'all','Lookback': Var.Lookback,'split': Var.split,
#      'batchsize': Var.batchsize,'Epochs': Var.Epochs,
#      'hidden_units':Var.hidden_units, 'Dens_unit': Var.Dense_Unit,
#      'dropout': Var.dropout, 'Kernel Regularizer': Var.Kr , 'Activity regularizer': Var.Ar,
#      'learning_rate': Var.learning_rate,'learning_rate_decay': Var.learning_rate_decay, 
#      'fold': Var.fold, 'Scale': Var.scalerange,'Loss_Function': Var.Loss_Function,
#      'Perf_Metric': Var.Perf_Metric,'Activation_funtion': Var.activationF,
#      'ResidualBlocks':Var.residual_blocks,'model' :Var.model, 'earlystoppingpatience': Var.early_stopping_patience,
#      'Metric weighting method':Var.Jmethod}


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
np.random.seed(42)
# How to pickle correct. 
# First initiate the class with instace attributes (self.a=[]) in the same file/function where you want to pickle. As only the instance attributes are pickled correctly
# The instanciate the class with Rrs=Results(). Otherwise there is no instances to pickle
# Then fill the instance with values
# https://stackoverflow/questions/10842553/pickle-with-custom-classes
# https://stackoverlfo/questions/16680221/pyhton-function-not-accessing-class-variable
class Results:
    def __init__(self):
       self.Info=Var.info
       self.Fold=Var.fold
       
       self.train_metric=[]
       self.train_loss=[]
       self.val_metric=[]
       self.val_loss=[]  
#    
       self.val_f1=[]
       self.val_k=[]
       self.val_recall=[]
       self.val_precision=[]
       self.val_no_mask_acc=[]
       self.train_f1=[]
       self.train_k=[]
       self.train_recall=[]
       self.train_precision=[]
       self.train_no_mask_acc=[] 
       
       self.mean_val_metric=[]
       self.mean_train_metric=[]
       self.mean_val_loss=[]
       self.mean_train_loss=[]
       
       self.mean_val_f1=[]
       self.mean_val_k=[]
       self.mean_val_recall=[]
       self.mean_val_precision=[]
       self.mean_val_no_mask_acc=[]
       self.mean_train_f1=[]
       self.mean_train_k=[]
       self.mean_train_recall=[]
       self.mean_train_precision=[]
       self.mean_train_no_mask_acc=[]
       
Ergebnisse=Results()     

  
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

model,Ergebnisse=leave_one_out_cross_validation(babies,AnnotMatrix_each_patient,FeatureMatrix_each_patient,Var,Varplus,Ergebnisse)
  
#SAVING STUFF
if Var.saving_model:
       #info https://stackoverflow.com/questions/42763094/how-to-save-final-model-using-keras
       from keras.models import model_from_json
       from keras.models import load_model
       
       def save_Model(model_json, filename):
              with open(filename, "w") as json_file:
                     json_file.write(model_json)
        
       model_json = model.to_json()
       save_Model(model_json,Var.resultpath+'/'+Var.runningNumber+'_'+Var.description+".json")   

       # serialize weights to HDF5
#       model.save_weights(Var.resultpath+'/'+Var.runningNumber+'_'+Var.description+'_weigths.h5')  # creates a HDF5 file 'my_model.h5' for the last used weights. Better checkpooint weights
       print('Model saved')
#
if Var.SavingResults:
       def save_object(obj, filename):
              with open(filename, 'wb') as output:  # Overwrites any existing file.
                     pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
       save_object(Ergebnisse, Var.resultpath+'/'+Var.runningNumber+'_'+Var.description+'_Ergebnisse.pkl' )  

       print('Results saved')

"""
END
"""
disp('-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - ')
   
   
import time
t=time.localtime()
zeit=time.asctime()
Minuten=(time.time() - start_time)/60
Stunden=(time.time() - start_time)/3600
print('FINISHED ' + Var.runningNumber + Var.description )
#print ('Runtime ' + runtime)
print("--- %i seconds ---" % (time.time() - start_time))
print("--- %i min ---" % Minuten)
print("--- %i h ---" % Stunden)

# Fill these in with the appropriate info...
usr='ScriptCallback@gmail.com'
psw='$Siegel#1'
fromaddr='ScriptCallback@gmail.com'
toaddr='jan.werth@philips.com'

# Send notification email
noticeEMail(starttime, usr, psw, fromaddr, toaddr,Var.runningNumber,Var.description)
#disp(  RES1_Kappa[-1])
#disp(RES1_kappa_STD)
#disp (RES1_KAPPA_overall)
