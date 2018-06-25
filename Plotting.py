# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:35:06 2018

@author: 310122653
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


XAxse= range(len(mean_train_loss))


train,=plt.plot(XAxse,mean_train_loss,label='Train',color=[0.4,0.4,0.4])
val,=plt.plot(XAxse,mean_val_loss,label='Val',color=[0,0,0])
plt.legend(handles=[train, val])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show

#plt.figure()
train,=plt.plot(XAxse,mean_train_metric,label='Train')#,color=[0.4,0.4,0.4])
val,=plt.plot(XAxse,mean_val_metric,label='Val')#,color=[0,0,0])
plt.legend(handles=[train, val])
plt.xlabel('Epochs')
plt.ylabel('Cat-acc')
plt.show

