# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:21:53 2018

@author: 310122653
"""
"""OWN Metrics.
"""
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import sys

class f1_precicion_recall_acc(Callback):
       def on_train_begin(self, logs={}):
              self.val_f1s = []
              self.val_recalls = []
              self.val_precisions = []
              self.val_accuracy=[]
 
       def on_epoch_end(self, epoch, logs={}):
              val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
              val_targ = self.model.validation_data[1]
              _val_f1 = f1_score(val_targ, val_predict)
              _val_recall = recall_score(val_targ, val_predict)
              _val_precision = precision_score(val_targ, val_predict)
              _val_accuracy = accuracy_score(val_targ, val_predict)
              self.val_f1s.append(_val_f1)
              self.val_recalls.append(_val_recall)
              self.val_precisions.append(_val_precision)
              self.val_accuracy.append(_val_accuracy)

              print ("  val_f1: %f  val_precision: %f  val_recall %f " %(_val_f1, _val_precision, _val_recall))
              return
 
class callbackexample(Callback):
       def on_train_begin(self, logs={}):
              self.train_accuracy=[]
              self.val_accuracy=[]
 
       def on_epoch_end(self, epoch, logs={}):
              train_predict= (np.asarray(self.model.predict(self.model.x))).round()
              train_targ= self.model.y 
              
              val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
              val_targ = self.model.validation_data[1] 

              _train_accuracy = accuracy_score(train_targ, train_predict)
              _val_accuracy = accuracy_score(val_targ, val_predict)
        
              self.train_accuracy.append(_train_accuracy)
              self.val_accuracy.append(_val_accuracy)
              print (" train_accuracy %f  val_accuracy %f" %(_train_accuracy, _val_accuracy))
              return
 
       
class categorical_accuracy_no_mask(Callback):  
       
       def on_train_begin(self, logs={}):
              self.train_acc = []
              self.val_acc = []
              

       def on_train_end(self, logs={}):
              return              
 
       def on_epoch_end(self, epoch, logs={}):
              train_predict= (np.asarray(self.model.predict(self.model.X_train_jan))).round()              
              val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
              train_targ= self.model.y[1] 
              val_targ = self.model.validation_data[1]
              indx_train=np.where(~train_targ.any(axis=2))[0] #find where all targets are zero. That are the masked once as we masked the target with 0 and the data with 666        
              indx_val=np.where(~val_targ.any(axis=2))[0] #find where all targets are zero. That are the masked once as we masked the target with 0 and the data with 666
              y_train_true_nomask = numpy.delete(train_targ, indx_train, axis=0)
              y_train_pred_nomask = numpy.delete(train_predict, indx_train, axis=0)        
              y_val_true_nomask = numpy.delete(val_targe, indx_val, axis=0)
              y_val_pred_nomask = numpy.delete(val_predict, indx_val, axis=0)
              equal(argmax(y_true_nomask, axis=-1), argmax(y_pred_nomask, axis=-1)) 

              _train_accuracy = accuracy_score(y_train_true_nomask, y_train_pred_nomask)            
              _val_accuracy = accuracy_score(y_true_nomask, y_pred_nomask)
              self.train_acc.append(_train_accuracy)        
              self.val_acc.append(_val_accuracy)

              print ("  train_acc: %f  val_acc: %f " %(_train_accuracy, _val_accuracy))
              return

class f1_prec_rec_acc_noMasking(Callback):
       def on_train_begin(self, logs={}):
              self.val_f1s = []
              self.val_recall = []
              self.val_precision = []
              self.val_accuracy=[]
              self.train_f1s = []
              self.train_recall = []
              self.train_precision = []
              self.train_accuracy=[]
 
       def on_epoch_end(self, epoch, logs={}):
              #LOAD DATA
              train_predict= (np.asarray(self.model.predict(self.model.X_train_jan)))    
              train_true=self.model.Y_train_jan           
              val_predict = np.asarray(self.model.predict(self.validation_data[0]))
              val_true = self.validation_data[1] 
              
              #MERGE ALL
              train_predict_merged=np.reshape(train_predict, (-1, np.shape(train_predict)[2])) # create a 2 dimensional array where all samples(first dimesion) are copied after each other onto the second dimension. The thirs is the label length(one hot encoded). -1 automatically creates the rigt length for dim 2 
              train_true_merged=np.reshape(train_true, (-1, np.shape(train_true)[2])) 
              val_predict_merged=np.reshape(val_predict, (-1, np.shape(val_predict)[2])) 
              val_true_merged=np.reshape(val_true, (-1, np.shape(val_true)[2])) 
              
              #DELETE MASKED VALUES
              indxT=np.where(~train_true_merged.any(axis=1))[0] #find where all targets are zero. That are the masked once as we masked the target with 0 and the data with 666
              indxV=np.where(~val_true_merged.any(axis=1))[0] #find where all targets are zero. That are the masked once as we masked the target with 0 and the data with 666
              
              train_true_nomask = np.delete(train_true_merged, indxT, axis=0)
              train_pred_nomask = np.delete(train_predict_merged, indxT, axis=0)               
              val_true_nomask = np.delete(val_true_merged, indxV, axis=0)                     
              val_pred_nomask = np.delete(val_predict_merged, indxV, axis=0) 
              #TURN ONE HOT ENCODED       
              train_true_nomask=np.argmax(train_true_nomask,axis=1) # F1 etc cannot handle one hot encoded targets
              train_pred_nomask=np.argmax(train_pred_nomask,axis=1)
              val_true_nomask=np.argmax(val_true_nomask,axis=1)
              val_pred_nomask=np.argmax(val_pred_nomask,axis=1)
              #CALC. PERFORMANCE
              _train_f1=               f1_score(train_true_nomask, train_pred_nomask,labels=self.model.label, pos_label=1, average=self.model.Jmethod)
              _train_recall=       recall_score(train_true_nomask, train_pred_nomask,labels=self.model.label, pos_label=1, average=self.model.Jmethod)
              _train_precision= precision_score(train_true_nomask, train_pred_nomask,labels=self.model.label, pos_label=1, average=self.model.Jmethod)
              _train_accuracy=   accuracy_score(train_true_nomask, train_pred_nomask,normalize= True)              
              
              _val_f1=               f1_score(val_true_nomask, val_pred_nomask,labels=self.model.label, pos_label=1, average=self.model.Jmethod)
              _val_recall=       recall_score(val_true_nomask, val_pred_nomask,labels=self.model.label, pos_label=1, average=self.model.Jmethod)
              _val_precision= precision_score(val_true_nomask, val_pred_nomask,labels=self.model.label, pos_label=1, average=self.model.Jmethod)
              _val_accuracy=   accuracy_score(val_true_nomask, val_pred_nomask, normalize= True)
              
              self.train_f1s.append(_val_f1)
              self.train_recall.append(_val_recall)
              self.train_precision.append(_val_precision)
              self.train_accuracy.append(_val_accuracy)
              
              self.val_f1s.append(_val_f1)
              self.val_recall.append(_val_recall)
              self.val_precision.append(_val_precision)
              self.val_accuracy.append(_val_accuracy)

              print (" train_accuracy: %f  val_f1: %f  val_precision: %f  val_recall: %f  _val_accuracy: %f" %(_train_accuracy, _val_f1, _val_precision, _val_recall, _val_accuracy))
              return
 
