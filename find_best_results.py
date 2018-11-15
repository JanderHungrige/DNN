#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:24:57 2018

@author: 310122653
"""

import numpy as np
import pickle
from keras.models import model_from_json

class Results:      
    Info=[]
    pass


system='Cluster' 
name='724_Res_Bi_ASQS_Kr_0.001_Ar_0.001_drop_0.5_wID_0_Ergebnisse'
Fold=1

if system=='Philips':
    pfad='C:/Users/310122653/Documents/PhD/Article_4_(MMC)/Results/'
if system=='Cluster':
    pfad='/home/310122653/Git/DNN/Results/'    
    pfad='/home/310122653/Git/DNN/Results_paper/'


#with (open(pfad+name+'_Ergebnisse.pkl', 'rb') ) as input:
#    Ergebniss=pickle.load(input)

with (open(pfad+name+'.pkl', 'rb') ) as input:
    Ergebnisse=pickle.load(input)  
#    
#with (open(pfad+name+'_Ergebnisse.pkl', 'rb') ) as input:
#    Ergebnisse=pickle.load(input)
#maxAcc=np.max(Ergebnisse.val_metric)
##maxf1=np.max(Ergebnisse.val_f1)
#maxk=np.max(Ergebnisse.val_k)
#maxkACC=np.ravel(Ergebnisse.val_metric)[Ergebnisse.val_k.index(maxk)]

#------------------------------------------

if Fold>0:
    
    a=np.max(Ergebnisse.val_metric[0,:])
    b=np.max(Ergebnisse.val_metric[1,:])
    c=np.max(Ergebnisse.val_metric[2,:])
    meanMaxAcc1=[np.mean([a,b,c]),np.std([a,b,c])]
    meanMaxAcc2=[np.mean([a,c]),np.std([a,c])]
    meanMaxAcc3=[np.mean([a,b]),np.std([a,b])]
    meanMaxAcc4=[np.mean([b,c]),np.std([b,c])]
    
    a=np.max(Ergebnisse.val_k[0,:])
    b=np.max(Ergebnisse.val_k[1,:])
    c=np.max(Ergebnisse.val_k[2,:])
    meanMaxk1=[np.mean([a,b,c]),np.std([a,b,c])]
    meanMaxk2=[np.mean([a,c]),np.std([a,c])]
    meanMaxk3=[np.mean([a,b]),np.std([a,b])]
    meanMaxk4=[np.mean([b,c]),np.std([b,c])]

#    del a,b,c,
    
del name, pfad, system, Fold