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
class Ergebnisse:
     def i(self):  
         pass
#Ergebnisse=Results()
class Results:
     def i(self):  
         pass
Ergebnisse=pickle.load(open("C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/24_AS_IS_MMCcECG_4GRU.pkl", "rb") )
 
# load json and create model
json_file = open('C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/11_All_MMC_3advmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/11_All_MMC_3advmodel_weigths.h5")
print("Loaded model from disk")
# 
fold=3
if fold>1:
       XAxse= range(len(Ergebnisse.mean_train_loss_overall))
       mean_train_metric=Ergebnisse.mean_train_metric_overall.T
       mean_val_metric=Ergebnisse.mean_val_metric_overall
       mean_train_loss=Ergebnisse.mean_train_loss_overall
       mean_val_loss=Ergebnisse.mean_val_loss_overall
       
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
       XAxse= range(len(Ergebnisse.mean_train_loss[0]))
      
       mean_train_metric_all=np.asarray(Ergebnisse.mean_train_metric)
       mean_val_metric_all=Ergebnisse.mean_val_metric
       mean_train_loss_all=Ergebnisse.mean_train_loss
       mean_val_loss_all=Ergebnisse.mean_val_loss
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
       
       plt.figure()
       train,=plt.plot(XAxse,mean_train_loss_all[i],label='Train',color=[0.4,0.4,0.4])
       val,=plt.plot(XAxse,mean_val_loss_all[i],label='Val',color=[0,0,0])
       plt.legend(handles=[train, val])
       plt.xlabel('Epochs')
       plt.ylabel('Loss')
       plt.show

print(Ergebnisse.info)
