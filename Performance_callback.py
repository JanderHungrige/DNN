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

              print (" — val_f1: %f — val_precision: %f — val_recall %f " %(_val_f1, _val_precision, _val_recall))
              return
 
class f1_prec_rec_acc_noMasking(Callback):
       def on_train_begin(self, logs={}):
              self.val_f1s = []
              self.val_recalls = []
              self.val_precisions = []
              self.val_accuracy=[]
 
       def on_epoch_end(self, epoch, logs={}):
              val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
              val_targ = self.model.validation_data[1] 
              indx=np.where(~val_targ.any(axis=2))[0] #find where all targets are zero. That are the masked once as we masked the target with 0 and the data with 666
              y_true_nomask = numpy.delete(val_targe, indx, axis=0)
              y_pred_nomask = numpy.delete(val_predict, indx, axis=0)
        
              _val_f1 = f1_score(y_true_nomask, y_pred_nomask)
              _val_recall = recall_score(y_true_nomask, y_pred_nomask)
              _val_precision = precision_score(y_true_nomask, y_pred_nomask)
              _val_accuracy = accuracy_score(y_true_nomask, y_pred_nomask)
        
              self.val_f1s.append(_val_f1)
              self.val_recalls.append(_val_recall)
              self.val_precisions.append(_val_precision)
              self.val_accuracy.append(_val_accuracy)

              print (" — val_f1: %f — val_precision: %f — val_recall %f - __val_accuracy %f" %(_val_f1, _val_precision, _val_recall, _val_accuracy))
              return
 
       
class categorical_accuracy_no_mask(Callback):
       
       
       def on_train_begin(self, logs={}):
              self.train_acc = []
              self.val_acc = []
              

	   def on_train_end(self, logs={}):
              return              
 
       def on_epoch_end(self, epoch, logs={}):
              train_predict= (np.asarray(self.model.predict(self.model.x[0]))).round()              
              val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
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

              print (" — train_acc: %f — val_acc: %f " %(_train_accuracy, _val_accuracy))
              return


