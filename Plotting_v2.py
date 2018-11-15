#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:18:07 2018

@author: 310122653
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:35:06 2018

@author: 310122653
"""

import numpy as np
import keras as K
import matplotlib.pyplot as plt
import pickle
from keras.models import model_from_json

#import wrapper_DNN
#from wrapper_DNN import Results
plt.style.use('ggplot')

  
       
#Ergebnisse=Results()
#class Ergebniss:
#     def I(self):  
#         pass
#Ergebnisse=Results()
class Results:      
    Info=[]
    pass
##  
  
system='Cluster' 
#name='3_Bi_ASQS_lb1337_7525_Kr0001Ar0001_drop04_w0_L1L2 lessbatchnorm_Ergebnisse'
name='724_Res_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Ergebnisse'


#name='1_Bi_ASQS_lb1337_7525_Kr0001Ar0001_drop04_w0_L1L2 lessbatchnorm_Ergebnisse'

saving=1
fold=1
print(name)
if system=='Philips':
    pfad='C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/'
if system=='Cluster':
    pfad='/home/310122653/Git/DNN/Results_paper/'
#    pfad='/home/310122653/Git/DNN/Results/'
#    pfad='/home/310122653/Git/DNN/Results/before7.11/'
    
    picpfad='/home/310122653/Git/DNN/Pictures/'

#with (open(pfad+name+'_Ergebnisse.pkl', 'rb') ) as input:
#    Ergebniss=pickle.load(input)
with (open(pfad+name+'.pkl', 'rb') ) as input:
    Ergebnisse=pickle.load(input)    
# load json and create model
#json_file = open(pfad +name+'.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights(pfad +name+"Fold_"+str(Fold)+"_checkpointbestmodel.hdf5")
#print("Loaded model from disk")
# 
    
if pfad=='/home/310122653/Git/DNN/Results_paper/':
    fold=2

if fold>1:
       XAxse= range(len(Ergebnisse.mean_train_loss))

       a=(Ergebnisse.val_k[0,:])
       b=(Ergebnisse.val_k[1,:])
       c=(Ergebnisse.val_k[2,:])
       meanMaxk1=[np.mean([a,b,c],0),np.std([a,b,c],0)]
       meanMaxk2=[np.mean([a,c],0),np.std([a,c],0)]
       meanMaxk3=[np.mean([a,b],0),np.std([a,b],0)]
       meanMaxk4=[np.mean([b,c],0),np.std([b,c],0)]        
 
       standardt=np.std(Ergebnisse.train_metric,0)
       standardv=np.std(Ergebnisse.val_metric,0)
#Plottin THE MEAN METRIC       
       train=plt.errorbar(XAxse,Ergebnisse.mean_train_metric,standardt,errorevery=100,ecolor=[0.4,0.4,0.4],capsize=5,label='Train',color=[0.4,0.4,0.4])
       val=plt.errorbar(XAxse,Ergebnisse.mean_val_metric,standardv,errorevery=100,ecolor=[0,0,0],capsize=5,solid_capstyle='projecting',label='Val',color=[0,0,0])
       plt.legend(handles=[train, val])
       plt.xlabel('Epochs')
       plt.ylabel('Cat-acc')
       plt.show
       
#       if saving:
#           plt.savefig(picpfad+name+'_metric.jpeg')
                    
#Plotting THE LOSS CAPED AT 2*MAX VAL LOSS       
       plt.figure()
       train,=plt.plot(XAxse,Ergebnisse.mean_train_loss,label='Train',color=[0.4,0.4,0.4])
       val,=plt.plot(XAxse,Ergebnisse.mean_val_loss,label='Val',color=[0,0,0])
       plt.legend(handles=[train, val])
       plt.xlabel('Epochs')
       plt.ylabel('Loss')
       plt.ylim(None,np.max(Ergebnisse.mean_val_loss)*2)
       plt.show
#       if saving:
#           plt.savefig(picpfad+name+'_loss.jpeg')   
         
#pLOTTING ALL  VALIDATION FOLDS       
       plt.figure()
       for i in range(len(Ergebnisse.val_metric)):
#              train,=plt.plot(XAxse,mean_train_metric_all[i].T,label='Train')#,color=[0.4,0.4,0.4])
#              val,=plt.plot(XAxse,Ergebnisse.val_metric[i],label='Val')#,color=[0,0,0])
              val,=plt.plot(XAxse,Ergebnisse.val_k[i],label='Val')
              plt.legend(handles=[val])
              plt.xlabel('Epochs')
              plt.ylabel('Cat-acc')
              plt.ylabel('Kappa')
#pLOTTING ALL TRAIN FOLDS              
       plt.figure()
       for i in range(len(Ergebnisse.val_metric)):
              train,=plt.plot(XAxse,Ergebnisse.train_metric[i],label='Train')#,color=[0.4,0.4,0.4])
       #       val,=plt.plot(XAxse,mean_val_metric_all[i],label='Val')#,color=[0,0,0])
              plt.legend(handles=[train])
              plt.xlabel('Epochs')
              plt.ylabel('Cat-acc')       
       plt.show
       
       
#All In subplot  
       #Accuray
       from matplotlib import gridspec 
       gs=gridspec.GridSpec(3,1)
       fig=plt.figure()
       ax=fig.add_subplot(gs[:-1,:])
       ax.errorbar(XAxse,Ergebnisse.mean_train_metric,standardt,errorevery=100,ecolor=[0.4,0.4,0.4],capsize=5,label='Train',color=[0.4,0.4,0.4])
       ax.errorbar(XAxse,Ergebnisse.mean_val_metric,standardv,errorevery=100,ecolor=[0,0,0],capsize=5,solid_capstyle='projecting',label='Val',color=[0,0,0])
       ax.set_ylabel('Cat-acc')
       ax.tick_params(axis='x',bottom='off',labelbottom='off') #only use one x axis
       
       ax2=fig.add_subplot(gs[-1,:])
       ax2.plot(XAxse,Ergebnisse.mean_train_loss,label='Train',color=[0.6,0.6,0.6])
       ax2.plot(XAxse,Ergebnisse.mean_val_loss,label='Val',color=[0.3,0.3,0.3])       
       ax2.set_ylabel('Loss')
       ax2.set_xlabel('Epochs')
       ax2.set_ylim(None,np.max(Ergebnisse.mean_val_loss)*2)
       ax.legend(loc=0)
       plt.show  
       
#       if saving:
#           plt.savefig(picpfad+name+'.jpeg') 
#           plt.savefig(picpfad+name+'.svg')
       
       #KAPPA
       standardt=np.std(Ergebnisse.train_k,0)
       standardv=np.std(Ergebnisse.val_k,0)          
       fig=plt.figure()
       ax=fig.add_subplot(gs[:-1,:])       
       ax.errorbar(XAxse,Ergebnisse.mean_train_k,standardt,errorevery=100,ecolor=[0.4,0.4,0.4],capsize=5,label='Train',color=[0.4,0.4,0.4])
       ax.errorbar(XAxse,Ergebnisse.mean_val_k,standardv,errorevery=100,ecolor=[0,0,0],capsize=5,solid_capstyle='projecting',label='Val',color=[0,0,0])
       ax.legend(loc=0)
       ax.set_ylabel('Kappa') 
       ax.tick_params(axis='x',bottom='off',labelbottom='off') #only use one x axis       

       ax2=fig.add_subplot(gs[-1,:])
       ax2.plot(XAxse,Ergebnisse.mean_train_loss,label='Train',color=[0.6,0.6,0.6])
       ax2.plot(XAxse,Ergebnisse.mean_val_loss,label='Val',color=[0.3,0.3,0.3])       
       ax2.set_ylabel('Loss')
       ax2.set_xlabel('Epochs')
       ax2.set_ylim(None,np.max(Ergebnisse.mean_val_loss)*2)
       ax.legend(loc=0)
       plt.show  
#---------1336
       standardt=np.std(Ergebnisse.train_k,0)
       standardv=meanMaxk3[1]         
       fig=plt.figure()
       ax=fig.add_subplot(gs[:-1,:])       
       ax.errorbar(XAxse,Ergebnisse.mean_train_k,standardt,errorevery=100,ecolor=[0.4,0.4,0.4],capsize=5,label='Train',color=[0.4,0.4,0.4])
       ax.errorbar(XAxse,meanMaxk3[0],standardv,errorevery=100,ecolor=[0,0,0],capsize=5,solid_capstyle='projecting',label='Val',color=[0,0,0])
       ax.legend(loc=0)
       ax.set_ylabel('Kappa') 
       ax.tick_params(axis='x',bottom='off',labelbottom='off') #only use one x axis       

       ax2=fig.add_subplot(gs[-1,:])
       ax2.plot(XAxse,Ergebnisse.mean_train_loss,label='Train',color=[0.6,0.6,0.6])
       ax2.plot(XAxse,Ergebnisse.mean_val_loss,label='Val',color=[0.3,0.3,0.3])       
       ax2.set_ylabel('Loss')
       ax2.set_xlabel('Epochs')
       ax2.set_ylim(None,np.max(Ergebnisse.mean_val_loss)*2)
       ax.legend(loc=0)
       plt.show         
       if saving:
           plt.savefig(picpfad+name+'.jpeg') 
           plt.savefig(picpfad+name+'.svg')
#All In ONE     
#       fig=plt.figure()
#       ax=fig.add_subplot(111)
#       ax.errorbar(XAxse,mean_train_metric,standardt,errorevery=100,ecolor=[0.4,0.4,0.4],capsize=5,label='Train',color=[0.4,0.4,0.4])
#       ax.errorbar(XAxse,mean_val_metric,standardv,errorevery=100,ecolor=[0,0,0],capsize=5,solid_capstyle='projecting',label='Val',color=[0,0,0])
#       ax.set_ylabel('Cat-acc')
#       ax2=plt.twinx()
#       ax2.plot(XAxse,mean_train_loss,label='Train',color=[0.6,0.6,0.6])
#       ax2.plot(XAxse,mean_val_loss,label='Val',color=[0.3,0.3,0.3])       
#       ax2.set_ylabel('Loss')
#       ax2.set_xlabel('Epochs')
#       ax2.set_ylim(None,np.max(mean_val_loss)*2)
#       ax2.legend(loc=0)
#       plt.show    

#PLOT FOR ONE FOLD OPERATIONS
if fold==1:
       XAxse= range(len(Ergebnisse.train_loss[0]))
      
       plt.figure()
       for i in range(len(Ergebnisse.train_metric)):
              train,=plt.plot(XAxse,Ergebnisse.train_metric[i],label='Train')#,color=[0.4,0.4,0.4])
              val,=plt.plot(XAxse,Ergebnisse.val_metric[i],label='Val')#,color=[0,0,0])
              plt.legend(handles=[ train, val])
              plt.xlabel('Epochs')
              plt.ylabel('Cat-acc')                  
       plt.show
       
       
       plt.figure()
       train,=plt.plot(XAxse,np.ravel(Ergebnisse.train_k),label='Train',color=[0.4,0.4,0.4])
       val,=plt.plot(XAxse,np.ravel(Ergebnisse.val_k),label='Val',color=[0,0,0])
       plt.legend(handles=[train, val])
       plt.xlabel('Epochs')
       plt.ylabel('Kappa')
       plt.show       
       
       plt.figure()
       train,=plt.plot(XAxse,Ergebnisse.train_loss[i],label='Train',color=[0.4,0.4,0.4])
       val,=plt.plot(XAxse,Ergebnisse.val_loss[i],label='Val',color=[0,0,0])
       plt.legend(handles=[train, val])
       plt.xlabel('Epochs')
       plt.ylabel('Loss')
       plt.show

#KAPPA and Loss all in one
       from matplotlib import gridspec 
       gs=gridspec.GridSpec(3,1)
       fig=plt.figure()
       ax=fig.add_subplot(gs[:-1,:])       
       ax.plot(XAxse,np.ravel(Ergebnisse.train_k),label='Train',color=[0.4,0.4,0.4])
       ax.plot(XAxse,np.ravel(Ergebnisse.val_k),label='Val',color=[0,0,0])
       ax.set_ylabel('Kappa')
       ax.tick_params(axis='x',bottom='off',labelbottom='off') #only use one x axis       

       ax2=fig.add_subplot(gs[-1,:])
       ax2.plot(XAxse,ravel(Ergebnisse.train_loss),label='Train',color=[0.4,0.4,0.4])
       ax2.plot(XAxse,ravel(Ergebnisse.val_loss),label='Val',color=[0,0,0])       
       ax2.set_ylabel('Loss')
       ax2.set_xlabel('Epochs')
       ax2.set_ylim(None,np.max(Ergebnisse.val_loss)*2)
       ax.legend(loc=0)
       plt.show  
       if saving:
           plt.savefig(picpfad+name+'.jpeg') 
           plt.savefig(picpfad+name+'.svg')
#print(Ergebnisse.info)
