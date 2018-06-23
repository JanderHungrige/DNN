# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:08:23 2018

@author: R2D2
"""

info={'label':label,'Features':'all','Lookback':Lookback,'split':split,'batchsize':batchsize,'Epochs':Epochs,'hidden_units':hidden_units, 
      'dropout':dropout,'learning_rate':learning_rate,'learning_rate_decay':learning_rate_decay, 'fold':fold, 'model':'LSTM_model_3_advanced' }


savedic={'mean_train_metric':mean_train_metric,'mean_train_metric_pp':mean_train_metric_pp,'mean_val_metric':mean_val_metric,'mean_val_metric_pp':mean_val_metric_pp,
          'mean_train_loss':mean_train_loss,'mean_train_loss_pp':mean_train_loss_pp,'mean_val_loss':mean_val_loss,'mean_val_loss_pp':mean_val_loss_pp}


np.save(os.path.join("C:/Users/C3PO/Documents/GitHub/DNN/Results/" , "mean_train_metric"), mean_train_metric)
np.save(os.path.join("C:/Users/C3PO/Documents/GitHub/DNN/Results/" , "mean_train_metric_pp"), mean_train_metric_pp)
np.save(os.path.join("C:/Users/C3PO/Documents/GitHub/DNN/Results/" , "mean_val_metric"), mean_val_metric)
np.save(os.path.join("C:/Users/C3PO/Documents/GitHub/DNN/Results/" , "mean_val_metric_pp"), mean_val_metric_pp)

np.save(os.path.join("C:/Users/C3PO/Documents/GitHub/DNN/Results/" , "mean_train_loss"), mean_train_loss)
np.save(os.path.join("C:/Users/C3PO/Documents/GitHub/DNN/Results/" , "mean_train_loss_pp"), mean_train_loss_pp)
np.save(os.path.join("C:/Users/C3PO/Documents/GitHub/DNN/Results/" , "mean_val_loss"), mean_val_loss)
np.save(os.path.join("C:/Users/C3PO/Documents/GitHub/DNN/Results/" , "mean_val_loss_pp"), mean_val_loss_pp)

np.save(os.path.join("C:/Users/C3PO/Documents/GitHub/DNN/Results/" , "info"), info)

np.save(os.path.join("C:/Users/C3PO/Documents/GitHub/DNN/Results/" , "savedic"), savedic)

