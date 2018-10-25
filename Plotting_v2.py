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
name='000_TEST_NEW_RESULTS'
Fold=0


if system=='Philips':
    pfad='C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/'
if system=='Cluster':
    pfad='/home/310122653/Git/DNN/Results/'


#with (open(pfad+name+'_Ergebnisse.pkl', 'rb') ) as input:
#    Ergebniss=pickle.load(input)
with (open(pfad+name+'_Ergebnisse.pkl', 'rb') ) as input:
    Ergebnisse=pickle.load(input)    
# load json and create model
json_file = open(pfad +name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(pfad +name+"Fold_"+str(Fold)+"_checkpointbestmodel.hdf5")
print("Loaded model from disk")
# 
fold=1
if fold>1:
       XAxse= range(len(Ergebnisse.mean_train_loss))
       mean_train_metric=Ergebnisse.mean_train_metric.T
       mean_val_metric=Ergebnisse.mean_val_metric
       mean_train_loss=Ergebnisse.mean_train_loss
       mean_val_loss=Ergebnisse.mean_val_loss
       
       mean_train_f1=Ergebnisse.mean_train_f1
       mean_train_loss_overall=Ergebnisse.mean_train_loss
       mean_train_no_mask_acc= Ergebnisse.mean_train_no_mask_acc
       mean_train_precicion=Ergebnisse.mean_train_precicion
       mean_train_recall= Ergebnisse.mean_train_recall
       mean_val_f1=Ergebnisse.mean_val_f1
       mean_val_loss_overall= Ergebnisse.mean_val_loss
       mean_val_metric_overall=Ergebnisse.mean_val_metric
       mean_val_no_mask_acc=Ergebnisse.mean_val_no_mask_acc
       mean_val_precicion=Ergebnisse.mean_val_precicion
       mean_val_recall=Ergebnisse.mean_val_recall

       #XAxse= range(len(Result.mean_train_loss))
       
       mean_train_metric_all=np.asarray(Ergebnisse.mean_train_metric)
       mean_val_metric_all=Ergebnisse.mean_val_metric
       mean_train_loss_all=Ergebnisse.mean_train_loss
       mean_val_loss_all=Ergebnisse.mean_val_loss
       
       
       
       train,=plt.plot(XAxse,mean_train_metric,label='Train')#,color=[0.4,0.4,0.4])
       val,=plt.plot(XAxse,mean_val_metric,label='Val')#,color=[0,0,0])
       plt.legend(handles=[train, val])
       plt.xlabel('Epochs')
       plt.ylabel('Cat-acc')
       plt.show
       
       plt.figure()
       train,=plt.plot(XAxse,mean_train_loss,label='Train',color=[0.4,0.4,0.4])
       val,=plt.plot(XAxse,mean_val_loss,label='Val',color=[0,0,0])
       plt.legend(handles=[train, val])
       plt.xlabel('Epochs')
       plt.ylabel('Loss')
       plt.show
       
       plt.figure()
       for i in range(len(mean_train_metric_all)):
       #       train,=plt.plot(XAxse,mean_train_metric_all[i].T,label='Train')#,color=[0.4,0.4,0.4])
              val,=plt.plot(XAxse,mean_val_metric_all[i],label='Val')#,color=[0,0,0])
              plt.legend(handles=[val])
              plt.xlabel('Epochs')
              plt.ylabel('Cat-acc')
       plt.figure()
       for i in range(len(mean_train_metric_all)):
              train,=plt.plot(XAxse,mean_train_metric_all[i].T,label='Train')#,color=[0.4,0.4,0.4])
       #       val,=plt.plot(XAxse,mean_val_metric_all[i],label='Val')#,color=[0,0,0])
              plt.legend(handles=[train])
              plt.xlabel('Epochs')
              plt.ylabel('Cat-acc')       
       plt.show

if fold==1:
       XAxse= range(len(Ergebnisse.train_loss[0]))
      
       mean_train_metric_all=np.asarray(Ergebnisse.train_metric)
       mean_val_metric_all=Ergebnisse.val_metric
       mean_train_loss_all=Ergebnisse.train_loss
       mean_val_loss_all=Ergebnisse.val_loss
       plt.figure()
       for i in range(len(train_metric)):
       #       train,=plt.plot(XAxse,mean_train_metric_all[i].T,label='Train')#,color=[0.4,0.4,0.4])
              val,=plt.plot(XAxse,val_metric[i],label='Val')#,color=[0,0,0])
              plt.legend(handles=[val])
              plt.xlabel('Epochs')
              plt.ylabel('Cat-acc')
              
       plt.figure()
       for i in range(len(train_metric)):
              train,=plt.plot(XAxse,mean_train_metric[i].T,label='Train')#,color=[0.4,0.4,0.4])
       #       val,=plt.plot(XAxse,mean_val_metric_all[i],label='Val')#,color=[0,0,0])
              plt.legend(handles=[train])
              plt.xlabel('Epochs')
              plt.ylabel('Cat-acc')       
       plt.show
       
       plt.figure()
       train,=plt.plot(XAxse,train_loss[i],label='Train',color=[0.4,0.4,0.4])
       val,=plt.plot(XAxse,val_loss[i],label='Val',color=[0,0,0])
       plt.legend(handles=[train, val])
       plt.xlabel('Epochs')
       plt.ylabel('Loss')
       plt.show

print(Ergebnisse.info)
