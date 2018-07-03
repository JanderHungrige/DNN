# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:21:19 2018

@author: 310122653
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 22:21:39 2017

@author: 310122653
"""
#LOADING cECG Matlab data (Feature and annotation Matrices)

#def loadingMatrizen():
    #When importing a file, Python only searches the current directory, 
    #the directory that the entry-point script is running from, and sys.path 
    #which includes locations such as the package installation directory 
import scipy.io as sio
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from AnnotationChanger import AnnotationChanger
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_approximation import RBFSampler




def Loading_data_all(dataset, selectedbabies, lst, FeatureSet, Rpeakmethod, scaler, \
                            merge34, Movingwindow, preaveraging, postaveraging, exceptNOF, onlyNOF, FEAT,\
                            dispinfo,usedPC):
       

       """
       START *************************************************************************
       """
       if 'ECG'== dataset:
              if ux:
                     folder=('/home/310122653/Pyhton_Folder/cECG/Matrices/')
              else:
                     folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/Matrices/')
     
       if 'cECG'==dataset:
              if ux:
                     folder=('/home/310122653/Pyhton_Folder/cECG/cMatrices/')
              else:
                     folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/cMatrices/')
                     
       if 'MMC'== dataset:        
              if usedPC=='Cluster':
                     folder=('/home/310122653/Pyhton_Folder/DNN/Matrices/')
              if usedPC=='Philips':
                     folder=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Processed_data/DNN_Matrices/Matrices_Features/')              
              if usedPC=='c3po':
                     folder=('C:/Users/C3PO/Desktop/Processed data/DNN_Matrices/Matrices_Features/') 
              
       # ONLY 5 MIN FEATURES AND ANNOTATIONS
       dateien_each_patient="FeatureMatrix_","Annotations_" #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
#       windowlength="30"
       if 'ECG'== dataset or 'cECG'== dataset:
           Neonate_all='4','5','6','7','9','10','11','12','13'
       if 'MMC'== dataset:
           Neonate_all='1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22'          
       babies=[i for i in range(len(selectedbabies))]# return to main function
       
       Neonate=[(Neonate_all[i]) for i in selectedbabies];Neonate=tuple(Neonate)
       FeatureMatrix_each_patient_all=[0]*len(Neonate)
       AnnotMatrix_each_patient=[0]*len(Neonate)
       t_a=[0]*len(Neonate)
       
       # IMPORTING *.MAT FILES
       for j in range(len(dateien_each_patient)): # j=0 Features  j=1 Annotations
           for k in range(len(Neonate)):
#               Dateipfad=folder+dateien_each_patient[j]+Neonate[k]+"_win_"+windowlength+".mat" #Building foldername
               Dateipfad=folder+dateien_each_patient[j]+Neonate[k]+".mat" #Building foldername

               sio.absolute_import   
               matlabfile=sio.loadmat(r'{}'.format(Dateipfad)) 
           
       # REWRITING FEATURES AND ANNOTATIONS    
           #NANs should already be deleted. Not scaled.
           #NANs can be in as there are only NaNs with NaN annotations. Nan al label is not used
               if j==0:
                   FeatureMatrix_each_patient_all[k]=matlabfile.get('FeatureMatrix')       
                   
               elif j==1:
                   AnnotMatrix_each_patient[k]=matlabfile.get('Annotations')  
                   AnnotMatrix_each_patient[k]=AnnotMatrix_each_patient[k].transpose() # transpose to datapoints,annotations
                   t_a[k]=np.linspace(0,len(AnnotMatrix_each_patient[k])*30/60,len(AnnotMatrix_each_patient[k]))  
#                   if plotting:
#                        plt.figure(k) 
#                        plt.plot(t_a[k],AnnotMatrix_each_patient[k])
#                        plt.title([k])
       #            AnnotMatrix_each_patient[k]= np.delete(AnnotMatrix_each_patient[k],(1,2), axis=1) #Reduce AnnotationMatrix to Nx1
       #            AnnotMatrix_each_patient[k]=AnnotMatrix_each_patient[k][~np.isnan(AnnotMatrix_each_patient[k]).any(axis=1)]#deleting NAN and turning Matrix to datapoints,Features
       
       if FeatureMatrix_each_patient_all[0].shape[0] < FeatureMatrix_each_patient_all[0].shape[1]: # We need [Data, Features]
                  FeatureMatrix_each_patient_all=[FeatureMatrix_each_patient_all[k].conj().transpose() for k in babies] # if the featuer matrix is turned wrongly, transpose it.                              
           
       if FeatureSet=='FET':
              FeatureMatrix_each_patient_all=[val[:,lst] for sb, val in enumerate(FeatureMatrix_each_patient_all)] # selecting only the features in lst                      
                                 
       if postaveraging:             
              NOF=np.arange(0,(np.size(FeatureMatrix_each_patient_all[K],1))) # create range from 0-29 (lenth of features)
              if exceptNOF:
                     NOF= np.delete(NOF,FEAT)
              if onlyNOF:
                     NOF=FEAT
              for F in NOF:#range(np.size(FeatureMatrix_each_patient_fromSession[K],1)):
                     FeatureMatrix_each_patient_all[K][:,F]=\
                     np.convolve(FeatureMatrix_each_patient_all[K][:,F], np.ones((Movingwindow,))/Movingwindow, mode='same')                
                                          
      
       
       AnnotMatrix_each_patient=AnnotationChanger(AnnotMatrix_each_patient,0,0,0,0,0,0,merge34)
       
# NORMALIZATION PER fEATURE (COLUMN) 
       if scaler==(0,1) or scaler==(-1,1):
              def Normaliz(values,scaler):
                     values=scaler.fit_transform(np.reshape(values,(-1,1)))
                     return values
             
              data =[0]*len(Neonate)         
              for matrix,i in zip(FeatureMatrix_each_patient_all,range(len(FeatureMatrix_each_patient_all))):  # iterate through every matrix in the list           
                  for column in matrix.transpose():  # iterate through every column in the matrix
                      NormCol=Normaliz(column,scaler) # call normalization function
                      if 'Matrx' in locals():
                             Matrx=np.hstack((Matrx,NormCol)) 
                      else:
                             Matrx=NormCol  
                  data[i]=Matrx 
                  del Matrx         
              FeatureMatrix_each_patient_all=data 
       
       
       return babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient_all
       
                     
                     
                     
       
       #%%
       
def Loading_data_perSession(dataset, selectedbabies, lst, FeatureSet, Rpeakmethod,ux, \
                            merge34, Movingwindow, preaveraging, postaveraging, exceptNOF, onlyNOF, FEAT,\
                            dispinfo,usedPC):    
             
       """
       Creating Feature Matrix per session
       """
       
       
       #folder=('/home/310122653/Pyhton_Folder/cECG/Matrices/')
       if 'ECG'== dataset:
              if ux:
                     Sessionfolder=('/home/310122653/Pyhton_Folder/cECG/Matrices/Sessions/')
              else:
                     Sessionfolder=('C:/Users/310122653/Dropbox/PHD/python/cECG/Matrices/Sessions/')
       if 'cECG'==dataset:
              if ux:
                     Sessionfolder=('/home/310122653/Pyhton_Folder/cECG/cMatrices/Sessions/')
              else:
                     Sessionfolder=('C:/Users/310122653/Dropbox/PHD/python/cECG/cMatrices/Sessions/')
       if 'MMC'== dataset:        
              if ux:
                     Sessionfolder=('/home/310122653/Pyhton_Folder/DNN/Matrices/Sessions/')
              else:
                     if usedPC=='Philips':
                            folder=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Processed_data/DNN_Matrices/Matrices_Features/')              
                     if usedPC=='c3po':
                            folder=('C:/Users/C3PO/Desktop/Processed data/DNN_Matrices/Matrices_Features/')       

 
       dateien_each_patient="FeatureMatrix_","Annotations_" #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
#       windowlength="300"
       if 'ECG'== dataset or 'cECG'== dataset:
           Neonate_all='4','5','6','7','9','10','11','12','13'
       if 'MMC'== dataset:
           Neonate_all='1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22'          
           
       babies=[i for i in range(len(selectedbabies))]# return to main function
       
       Neonate=[(Neonate_all[i]) for i in selectedbabies];Neonate=tuple(Neonate)
       FeatureMatrix_each_patient_all=[0]*len(Neonate)
       AnnotMatrix_each_patient=[0]*len(Neonate)
       t_a=[0]*len(Neonate)                     
#IMPORTING FILES
       import os
       import glob
       from pathlib import Path           
       
       FeatureMatrix_each_patient_fromSession=[None]*len(Neonate)
       FeatureMatrix_each_patient_fromSession_poly=[None]*len(Neonate)

       AnnotMatrix_each_patient=Loading_Annotations(dataset,selectedbabies,ux,usedPC,plotting=0) # loading Annotations
       
       for K in range(len(Neonate)):      
              Dateien=glob.glob(Sessionfolder +'FeatureMatrix_'+Neonate[K]+ '_**')
              FeatureMatrix_Session_each_patient=[None]*len(Dateien)


       
       # IMPORTING *.MAT FILES
              for j in range(len(Dateien)):
                     sio.absolute_import   
                     matlabfile=sio.loadmat(r'{}'.format(Dateien[j])) 
           
       # REWRITING FEATURES AND ANNOTATIONS    
           #NANs should already be deleted. Not scaled.
           #NANs can be in as there are only NaNs with NaN annotations. Nan al label is not used
                     FeatureMatrix_Session_each_patient[j]=matlabfile.get('FeatureMatrix') 
                     
#                      FeatureMatrix_Session_each_patient[j]=FeatureMatrix_Session_each_patient[j].transpose() # transpose to datapoints,features
       # TRIMMING THEM IF SESSIONS ARE TO SHORT OR EMPTY               
#              FeatureMatrix_Session_each_patient[1]=[];FeatureMatrix_Session_each_patient[5]=[]# just a test delete
              WelcheSindLeer=list()
              WelcheSindzuKurz=list()
              for j in range(len(Dateien)): 
                     if len(FeatureMatrix_Session_each_patient[j])==0:
                            WelcheSindLeer.append(j) #Just count how many Sessions do not have cECG values. If more than one different strategy is needed than the one below
                     if len(FeatureMatrix_Session_each_patient[j])!=0 and len(FeatureMatrix_Session_each_patient[j])<=2: # If a session is to short, remove it
                            WelcheSindzuKurz.append(j) #Just count how many Sessions do not have cECG values. If more than one different strategy is needed than the one below
                            
              if WelcheSindzuKurz:# deleting the ones that are too short in Annotation MAtrix. Apparently if there is no data(leer), then the Annotations are already shortened. Therefore only corretion for the once which are a bit to short
                     AnnotMatrix_each_patient=correcting_Annotations_length(dataset,K,WelcheSindzuKurz,ux,selectedbabies,AnnotMatrix_each_patient,FeatureMatrix_Session_each_patient)
                     WelcheSindLeer.extend(WelcheSindzuKurz)# delete the ones that are zero and the once that are too short
                     
              for index in sorted(WelcheSindLeer, reverse=True):
                     del FeatureMatrix_Session_each_patient[index]
#              FeatureMatrix_Session_each_patient=[m for n, m in enumerate(FeatureMatrix_Session_each_patient) if n not in WelcheSindLeer] #remove empty session


              FeatureMatrix_each_patient_fromSession[K]=np.concatenate(FeatureMatrix_Session_each_patient)
              
       if FeatureSet=='Features':             # only the Feature set has features to choose from :-) 
              FeatureMatrix_each_patient_fromSession=[val[:,lst] for sb, val in enumerate(FeatureMatrix_each_patient_fromSession)] # selecting only the features used iin lst
              
              
       for K in range(len(Neonate)):           
                 if postaveraging:             
                     NOF=np.arange(0,(np.size(FeatureMatrix_each_patient_fromSession[K],1))) # create range from 0-29 (lenth of features)
                     if exceptNOF:
                            NOF= np.delete(NOF,FEAT)
                     if onlyNOF:
                            NOF=FEAT
                     for F in NOF:#range(np.size(FeatureMatrix_each_patient_fromSession[K],1)):
                            FeatureMatrix_each_patient_fromSession[K][:,F]=\
                            np.convolve(FeatureMatrix_each_patient_fromSession[K][:,F], np.ones((Movingwindow,))/Movingwindow, mode='same')                
                     
       AnnotMatrix_each_patient=AnnotationChanger(AnnotMatrix_each_patient,LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,Smoothing_short,Pack4,direction6,merge34)
#                     
#       if plotting:
#              for l in range(len(AnnotMatrix_each_patient)): 
#                 t_a[l]=np.linspace(0,len(AnnotMatrix_each_patient[l])*30/60,len(AnnotMatrix_each_patient[l]))  
#                 plt.figure(l) 
#                 plt.plot(t_a[l],AnnotMatrix_each_patient[l]-0.1)
#                 plt.title([l])                            
                                    
                 
       return babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient_fromSession
#%%      
def Feature_names():

       Class_dict={1:'AS',2:'QS',3:'Wake', 4:'caretaking',5:'Unknown',6:'Trans'} #AS:active sleep   QS:Quiet sleep  AW:Active wake  QW:Quiet wake  Trans:Transition  Pos:Position         
       
       features_dict={
                      0:"Bpe",
                      1:"LineLength",
                      2:"meanLineLength",
                      3:"NN10",                  
                      4:"NN20",                
                      5:"NN30",                 
                      6:"NN50",                  
                      7:"pNN10",               
                      8:"pNN20",               
                      9:"pNN30",               
                      10:"pNN50",              
                      11:"RMSSD",
                      12:"SDaLL",                 
                      13:"SDANN", 
                      14:"SDLL",
                      15:"SDNN",
                      16:"pDEC",
                      17:"SDDEC",
                      18:"HF",                 
                      19:"HFnorm",
                      20:"LF",                  
                      21:"LFnorm",               
                      22:"ratioLFHF",           
                      23:"sHF",
                      24:"sHFnorm",
                      25:"totpower",
                      26:"uHF",
                      27:"uHFnorm",
                      28:"VLF",
                      29:"SampEN",
                      30:"QSE",
                      31:"SEAUC",                     
                      32:"LZNN",
                      33:"LZECG"
                      }                      
       features_indx = dict((y,x) for x,y in features_dict.items())       
       return Class_dict, features_dict, features_indx
#%%
def Loading_Annotations(dataset,selectedbabies,ux,usedPC,plotting):
       if 'ECG'== dataset:
              if ux:
                     folder=('/home/310122653/Pyhton_Folder/cECG/Matrices/')
              else:
                    folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/Matrices/')
       if 'cECG'==dataset:
              if ux:
                     folder=('/home/310122653/Pyhton_Folder/cECG/cMatrices/')
              else:
                     folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/cMatrices/')
       if 'MMC'== dataset:        
              if ux:
                     folder=('/home/310122653/Pyhton_Folder/DNN/Matrices/')
              else:
                     if usedPC=='Philips':
                            folder=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Processed_data/DNN_Matrices/Matrices_Features/')               
                     if usedPC=='c3po':
                            folder=('C:/Users/C3PO/Desktop/Processed data/DNN_Matrices/Matrices_Features/') 

           
       # ONLY 5 MIN FEATURES AND ANNOTATIONS
       dateien_each_patient="FeatureMatrix_","Annotations_" #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
#       windowlength="30"
       if 'ECG'== dataset or 'cECG'== dataset:
           Neonate_all='4','5','6','7','9','10','11','12','13'
       if 'MMC'== dataset:
           Neonate_all='1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22'          
           babies=[i for i in range(len(selectedbabies))]# return to main function
       
       Neonate=[(Neonate_all[i]) for i in selectedbabies];Neonate=tuple(Neonate)
       FeatureMatrix_each_patient_all=[0]*len(Neonate)
       AnnotMatrix_each_patient=[0]*len(Neonate)
       t_a=[0]*len(Neonate)
       
       # IMPORTING *.MAT FILES
       for k in range(len(Neonate)):
#               Dateipfad=folder+dateien_each_patient[j]+Neonate[k]+"_win_"+windowlength+".mat" #Building foldername
           Dateipfad=folder+dateien_each_patient[1]+Neonate[k]+ ".mat" #Building foldername
        
           sio.absolute_import   
           matlabfile=sio.loadmat(r'{}'.format(Dateipfad)) 
    
           AnnotMatrix_each_patient[k]=matlabfile.get('Annotations')  
           AnnotMatrix_each_patient[k]=AnnotMatrix_each_patient[k].transpose() # transpose to datapoints,annotations
           t_a[k]=np.linspace(0,len(AnnotMatrix_each_patient[k])*30/60,len(AnnotMatrix_each_patient[k]))  
#                   if plotting:
#                        plt.figure(k) 
#                        plt.plot(t_a[k],AnnotMatrix_each_patient[k])
#                        plt.title([k])
                        
       return AnnotMatrix_each_patient
       
#%%
def correcting_Annotations_missing(dataset,K,WelcheSindLeer,ux,selectedbabies,AnnotMatrix_each_patient,FeatureMatrix_Session_each_patient):
# This function is needed to load the ECG if the cECG Is loaded to compare the length of missing cECG value. The missing length can then be deleted from the annotations      
       if 'ECG'== dataset:
              if ux:
                     folder=('/home/310122653/Pyhton_Folder/cECG/Matrices/Sessions/')
              else:
                    folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/Matrices/Sessions/')
       if 'cECG'==dataset:
              if ux:
                     folder=('/home/310122653/Pyhton_Folder/cECG/cMatrices/Sessions/')
              else:
                     folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/cMatrices/Sessions/')
       if 'MMC'== dataset:        
              if ux:
                     folder=('/home/310122653/Pyhton_Folder/DNN/Matrices/Sessions/')
              else:
                     folder=('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Processed_data/DNN_Matrices/Matrices_Features//Sessions/') 


       # ONLY 5 MIN FEATURES AND ANNOTATIONS
       dateien_each_patient="FeatureMatrix_","Annotations_" #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
       if 'ECG'== dataset or 'cECG'== dataset:
           Neonate_all='4','5','6','7','9','10','11','12','13'
       if 'MMC'== dataset:
           Neonate_all='1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22'          
           babies=[i for i in range(len(selectedbabies))]# return to main function
       
       Neonate=[(Neonate_all[i]) for i in selectedbabies];Neonate=tuple(Neonate)
       FeatureMatrixECG=[0]*len(Neonate)
       
       import os
       import glob
       from pathlib import Path
             
       FeatureMatrix_each_patient_fromSession=[None]*len(Neonate)            
            
       Dateien=glob.glob(Sessionfolder +'FeatureMatrix_'+Neonate[K]+ '_**')
       FeatureMatrix_Session_each_patient=[None]*len(Dateien)

       
       # IMPORTING *.MAT FILES
       for w in range(len(Dateien)): 
               sio.absolute_import   
               matlabfile=sio.loadmat(r'{}'.format(Dateien[w])) 
    
       # REWRITING FEATURES AND ANNOTATIONS    
           #NANs should already be deleted. Not scaled.
           #NANs can be in as there are only NaNs with NaN annotations. Nan al label is not used
               FeatureMatrix_Session_each_patient[w]=matlabfile.get('FeatureMatrix') 
               FeatureMatrix_Session_each_patient[w]=FeatureMatrix_Session_each_patient[w].transpose() # transpose to datapoints,features
       
       # COllecting all the indices from the parts where single sessions are empty. We get the start index and the lengt. This is combined into on lare arry with all the indices that are missing. Those are then deleted at once from the annotations
       IndexRange=np.zeros(len(WelcheSindLeer))
       startindex=list()
       for l in range(len(WelcheSindLeer)):
              IndexRange[l]=(len(FeatureMatrix_Session_each_patient[WelcheSindLeer[l]])) # collect how many smaples are missing
              VonDa=[[(len(FeatureMatrix_Session_each_patient[t])) for t in range(WelcheSindLeer[l])]] # starting index by collecting all length before the missing one
              VonDa=np.sum(VonDa)
              startindex.append(list(range(VonDa+1,VonDa+1+np.int(IndexRange[l]))))# getting all the start indices in one array/list

       indices=np.hstack(startindex)
       AnnotMatrix_each_patient[K]= np.delete(AnnotMatrix_each_patient[K][:],[indices])

       return AnnotMatrix_each_patient
#%%
def correcting_Annotations_length(K,WelcheSindzuKurz,ux,selectedbabies,AnnotMatrix_each_patient,FeatureMatrix_Session_each_patient):
# This function is needed to load the ECG if the cECG Is loaded to compare the length of missing cECG value. The missing length can then be deleted from the annotations      

       # COllecting all the indices from the parts where single sessions are empty. We get the start index and the lengt. This is combined into on lare arry with all the indices that are missing. Those are then deleted at once from the annotations
       IndexRange=np.zeros(len(WelcheSindzuKurz))
       startindex=list()
       for l in range(len(WelcheSindzuKurz)):
              IndexRange[l]=(len(FeatureMatrix_Session_each_patient[WelcheSindzuKurz[l]])) # collect how many smaples are missing
              VonDa=[[(len(FeatureMatrix_Session_each_patient[t])) for t in range(WelcheSindzuKurz[l])]] # starting index by collecting all length before the missing one
              VonDa=np.int(np.sum(VonDa))
#              startindex.append(list(range(VonDa+1,VonDa+1+np.int(IndexRange[l]))))# getting all the start indices in one array/list
              startindex.append(list(range(VonDa,VonDa+np.int(IndexRange[l]))))# getting all the start indices in one array/list

       indices=np.hstack(startindex)
       indices=sorted(indices, reverse=True )
       
#       for i in range(len(indices)):
#              AnnotMatrix_each_patient[K]= np.delete(AnnotMatrix_each_patient[K],[indices[i],1])
       AnnotMatrix_each_patient[K]= np.delete(AnnotMatrix_each_patient[K][:],[indices])
       AnnotMatrix_each_patient[K]=AnnotMatrix_each_patient[K][:, None] # make it a 2D array. Otherwise error later
       return AnnotMatrix_each_patient


       