#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 22:25:47 2018

@author: 310122653
"""

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
#name=['900_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Ergebnisse.pkl',
#'1300_Res_MMC_Kr_0.0001_Ar_0.0001_drop_0.6_wID_0_Ergebnisse.pkl',
#'1300_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Ergebnisse.pkl',
#'1300_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Ergebnisse.pkl',
#'1600_Res_MMC_Kr_0.0001_Ar_0.0001_drop_0.6_wID_0_Ergebnisse.pkl',
#'1600_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Ergebnisse.pkl',
#'1600_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Ergebnisse.pkl',
#'1600_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Ergebnisse.pkl',
#'1800_Res_MMC_Kr_0.0001_Ar_0.0001_drop_0.6_wID_0_Ergebnisse.pkl',
#'1800_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Ergebnisse.pkl',
#'1800_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Ergebnisse.pkl',
#'1900_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
#'1900_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl', 
#'1900_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl', 
#'2600_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Ergebnisse.pkl',
#'2600_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Ergebnisse.pkl',
#
#'2900_Res_MMC_Kr_0.0001_Ar_0.0001_drop_0.6_wID_0_Ergebnisse.pkl',
#'2900_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Ergebnisse.pkl', 
#'2900_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Ergebnisse.pkl', 

#]

#name=['177_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_Ergebnisse.pkl',
#'177_Bi_ASQS_Kr_0.001_Ar_0.0_drop_0.0_Ergebnisse.pkl',
#'177_Bi_ASQS_Kr_0.01_Ar_0.001_drop_0.0_Ergebnisse.pkl',
#'177_Bi_ASQS_Kr_0.01_Ar_0.0_drop_0.5_Ergebnisse.pkl',
#'177_Bi_ASQS_Kr_0.0_Ar_0.0_drop_0.58020_Ergebnisse.pkl',
#'177_Bi_ASQS_Kr_0.0_Ar_0.0_drop_0.5_Ergebnisse.pkl',
#'177_Bi_ASQS_Kr_0.0_Ar_0.0_drop_0.6_Ergebnisse.pkl',
#'177_Bi_ASQS_Kr_0.1_Ar_0.0_drop_0.0_Ergebnisse.pkl',
#'177_Bi_ASQS_Kr_0.2_Ar_0.0_drop_0.0_Ergebnisse.pkl',
#]

#name=[
#'1_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'1_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_Ergebnisse.pkl',
#
#'1_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'1_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_Ergebnisse.pkl',
#
#'1_Bi_ASQS_Kr_0.0001_Ar_0.0001_drop_0.5_Ergebnisse.pkl',
#'1_Bi_ASQS_Kr_0.01_Ar_0.001_drop_0.3_Ergebnisse.pkl',
#'1_Bi_ASQS_Kr_0.01_Ar_0.001_drop_0.5_Ergebnisse.pkl',
#
#'2_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'2_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_Ergebnisse.pkl',
#'2_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'2_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_Ergebnisse.pkl',
#'2_Bi_ASQS_Kr_0.0001_Ar_0.0001_drop_0.5_Ergebnisse.pkl',
#'2_Bi_ASQS_Kr_0.01_Ar_0.001_drop_0.5_Ergebnisse.pkl',
#'2_Bi_ASQS_Kr_0.0001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
#
#'3_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'3_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_Ergebnisse.pkl',
#
#'3_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'3_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_Ergebnisse.pkl',
#
#'3_Bi_ASQS_Kr_0.0001_Ar_0.0001_drop_0.5_Ergebnisse.pkl',
#'3_Bi_ASQS_Kr_0.01_Ar_0.001_drop_0.5_Ergebnisse.pkl',
#'4_Bi_ASQS_Kr_0.0001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
#
#'4_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'4_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_Ergebnisse.pkl',
#
#'4_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'4_Bi_ASQS_Kr_0.01_Ar_0.001_drop_0.5_Ergebnisse.pkl',
#'5_Bi_ASQS_Kr_0.0001_Ar_0.0001_drop_0.5_Ergebnisse.pkl',
#'5_Bi_ASQS_Kr_0.0001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
#'5_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'5_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'5_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_Ergebnisse.pkl',
#'5_Bi_ASQS_Kr_0.01_Ar_0.001_drop_0.5_Ergebnisse.pkl',
#'6_Bi_ASQS_Kr_0.01_Ar_0.001_drop_0.5_Ergebnisse.pkl',
#'7_Bi_ASQS_Kr_0.01_Ar_0.001_drop_0.3_Ergebnisse.pkl',
#
#]
#

name=['1600_Res_MMC_Kr_0.001_Ar_0.001_drop_0.6_wID_0_Fold_1_Model_4_Ergebnisse.pkl',
'1800_Res_MMC_Kr_0.0001_Ar_0.0001_drop_0.6_wID_0_Ergebnisse.pkl',
'1800_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Ergebnisse.pkl',
'1800_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Ergebnisse.pkl',
'1900_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
'1900_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',

'2300_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Ergebnisse.pkl',
'2800_Res_MMC_Kr_0.0001_Ar_0.0001_drop_0.6_wID_0_Ergebnisse.pkl',
'2800_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Ergebnisse.pkl',
'2800_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Ergebnisse.pkl',
]
#name=[
#'1_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'1_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'2_Bi_ASQS_Kr_0.0001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
#'2_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'2_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'3_Bi_ASQS_Kr_0.0001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
#'3_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'3_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'4_Bi_ASQS_Kr_0.0001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
#'4_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'4_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'5_Bi_ASQS_Kr_0.0001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
#'5_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'5_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'6_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'6_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'7_Bi_ASQS_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'7_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#
#'5_all_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'5_all_Kr_0.01_Ar_0.001_drop_0.5_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#'5_all_Kr_0.01_Ar_0.01_drop_0.6_wID_0_Fold_1_Model_3_Ergebnisse.pkl',
#]

name=[
#'800_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Ergebnisse.pkl',
#'300_Res_MMC_Kr_0.0001_Ar_0.0001_drop_0.6_wID_0_Ergebnisse.pkl',
#'300_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Ergebnisse.pkl', 
#'306_Res_MMC_ECG_InSe_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
#'601_Res_ECG_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Ergebnisse.pkl',
#'600_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Ergebnisse.pkl', 
#'606_Res_MMC_ECG_InSe_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
'900_Res_MMC_Kr_0.001_Ar_0.0001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
'900_Res_MMC_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
'906_Res_MMC_ECG_InSe_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Fold_1_Ergebnisse.pkl',
]




#name='1_Bi_ASQS_lb1337_7525_Kr0001Ar0001_drop04_w0_L1L2 lessbatchnorm_Ergebnisse'

saving=0
fold=1
print(name)
if system=='Philips':
    pfad='C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/'
if system=='Cluster':
    pfad='/home/310122653/Git/DNN/Results_paper/'
    pfad='/home/310122653/Git/DNN/Results/'
#    pfad='/home/310122653/Git/DNN/Results/before7.11/'
    
    picpfad='/home/310122653/Git/DNN/Pictures/'

for i in range(len(name)):
    with (open(pfad+name[i], 'rb') ) as input:
        Ergebnisse=pickle.load(input)    

    XAxse= range(len(Ergebnisse.val_loss[0]))
      


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
    ax2.set_xlabel(name[i])
    ax2.set_ylim(None,np.max(Ergebnisse.val_loss)*2)
    ax.legend(loc=0)
    plt.show  
#if saving:
#    plt.savefig(picpfad+name+'.jpeg') 
#    plt.savefig(picpfad+name+'.svg')
#print(Ergebnisse.info)
