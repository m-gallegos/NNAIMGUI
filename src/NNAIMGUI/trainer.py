"""
Trainer module, allows the easy and fast creation of FFNN models
for the prediction of atomic properties.
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from NNAIMGUI import SFC

def clear_models():
  """
  Clear models after training
  """
  tf.keras.backend.clear_session()
  return None

def build_model(inshp,lr,loss,opt_type,neurons,activations):
  """
  Build the model architecture and return the model
  -inshp      : shape of the input layer
  -lr         : learning rate
  -loss       : loss function type
  -opt_type   : optimizer
  -neurons    : neurons of each layer in a list
  -activations: activation functions in a list
  """
  inputs = tf.keras.Input(shape=(inshp,))
  x = tf.keras.layers.Dense(int(neurons[0]), activation=activations[0])(inputs)
  for i in range(1,len(neurons)):
      x = tf.keras.layers.Dense(int(neurons[i]), activation=activations[i])(x)
  output  = tf.keras.layers.Dense(1)(x)
  model = tf.keras.Model(inputs, output)
  optimizer = eval(opt_type+"(learning_rate="+str(lr)+")")
  model.compile(loss=loss,
                optimizer=optimizer,
                metrics=['mae','mse'])
  return model

def norm(x,stats):
  """
  Normalize the data
  """
  val=(x - stats['mean'])/stats['std'] 
  # Check for NaN data
  try:
    assert not np.any(np.isnan(val))
  except AssertionError:
    raise( AssertionError( "Fatal error while normalizing the data, check your ACSF features for null sdev or mean! ") )
  return val

def show_model_performance(name,model,train_data,train_labels,test_data,test_labels):
  """
  Function which prints the model performance to an output file
  """
  # Evaluate the model on the training and testing data
  train_predictions = model.predict(train_data)
  test_predictions = model.predict(test_data)
  # Calculate the MAE and RMSE for the training and testing sets
  train_mae = np.mean(np.abs(train_predictions.ravel() - train_labels.ravel()))
  test_mae = np.mean(np.abs(test_predictions.ravel() - test_labels.ravel()))
  train_rmse = np.sqrt(np.mean(np.square(train_predictions.ravel() - train_labels.ravel())))
  test_rmse = np.sqrt(np.mean(np.square(test_predictions.ravel() - test_labels.ravel())))
  # Save the training and testing predictions
  with open(name+".train_pred", mode='w+') as file:
      file.write('# Training predictions with labels and metrics:\n')
      file.write('# MAE: {:.6f}\n'.format(train_mae))
      file.write('# RMSE: {:.6f}\n'.format(train_rmse))
      file.write('Predictions,Reference data\n')
      np.savetxt(name+".train_pred", np.concatenate([train_predictions, np.expand_dims(train_labels, axis=1)], axis=1), fmt='%.6e', delimiter=',',header='Predictions\tReference data')
  with open(name+".test_pred", mode='w+') as file:
      file.write('# Testing predictions with labels and metrics:\n')
      file.write('# MAE: {:.6f}\n'.format(test_mae))
      file.write('# RMSE: {:.6f}\n'.format(test_rmse))
      file.write('Predictions,Reference data\n')
      np.savetxt(name+".test_pred", np.concatenate([test_predictions, np.expand_dims(test_labels,axis=1)], axis=1), fmt='%.6e', delimiter=',',header='Predictions\tReference data')
  # Print the predictions, true labels, MAE, and RMSE
  print("Training MAE  : {:.6f}".format(train_mae))
  print("Training RMSE : {:.6f}".format(train_rmse))
  print("Testing  MAE  : {:.6f}".format(test_mae))
  print("Testing  RMSE : {:.6f}".format(test_rmse))
  return None

def set_database(dataset):
  """
  Set database and its statistics
  """
  data_stats = dataset.describe() 
  data_stats.pop("prop")
  data_stats = data_stats.transpose()
  data_labels=dataset.pop("prop")  
  return data_stats,data_labels

def train_from_csv(datafile="data.txt",ftra=0.8,vsplit=0.2,nepochs=100000,patnc=25,lr=0.000001,loss='mse',optimizer='RMSprop',neurons=(10,10,10),activations=('tanh','tanh','linear')):
  """
  Train an atomistic FFNN model from a database stored in a CSV file
  datafile   : name of the file containing the database in CSV format
  tfra       : fraction of the total data to be used for training
  vsplit     : fraction used for the validation split
  nepochs    : number of training EPOCHS
  patience   : patience used to prevent over-fitting
  lr         : learning rate 
  loss       : statistical metric used as loss function
  optimizer  : tf.keras.optimizers to be used
  neurons    : list containing the shape of the hidden layers
  activations: list with the activations of each layer
  """
  # Check the optimizer
  if (optimizer not in tf.keras.optimizers.__dict__.keys()):
     print("Error: Unrecognized optimizer ", optimizer)
     print(tf.keras.optimizers.__dict__.keys())
     exit(0)
  else:
     opt_type=str('tf.keras.optimizers.') + str(optimizer)
  # Check the activation functions
  for i in activations:
      if i not in tf.keras.activations.__dict__.keys():
         print("Error: Unrecognized activation ", i)
         print(tf.keras.activations.__dict__.keys())
         exit(0)
  # Get main path to file
  file_path=(os.path.dirname(datafile))
  if len(file_path)==0 : file_path=os.path.join(os.getcwd()) 
  # Get data from database
  with open(datafile,'r') as f:
    line = f.readline()
  inshp=len(line.split(','))-1
  rang    = range(1,inshp+1)
  colname = ['prop']
  for i in rang:
      colname.append('G' + str(i))
  raw_dataset = pd.read_csv(datafile,names=colname,na_values="?",
                            comment='\t',sep=",",skipinitialspace=True)
  dataset = raw_dataset.copy()
  dataset = dataset.dropna()
  # Split and normalize the database
  train_dataset = dataset.sample(frac=ftra,random_state=0)
  test_dataset = dataset.drop(train_dataset.index)
  train_stats,train_labels=set_database(train_dataset)
  test_stats,test_labels=set_database(test_dataset)
  normed_train_data = norm(train_dataset,train_stats)
  normed_test_data = norm(test_dataset,train_stats)   
  normed_train_stats = normed_train_data.describe()
  normed_train_stats = normed_train_stats.transpose()
  normed_test_stats = normed_test_data.describe()
  normed_test_stats = normed_test_stats.transpose()
  # Set the model
  class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
      if epoch % 100 == 0: print('')
      print('.', end='')
  model = build_model(inshp,lr,loss,opt_type,neurons,activations)
  model.summary()
  # Train the model
  checkpoint_dir = os.path.dirname(file_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                   save_weights_only=True,
                                                   verbose=1)
  early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patnc)
  history=model.fit(normed_train_data, train_labels, epochs=nepochs,
                      validation_split =vsplit, verbose=1, callbacks=[early_stop, PrintDot(),cp_callback])
  # Save the model
  model.save(os.path.join(file_path,datafile.split('.')[0]+".h5"))
  # Save the training statistics
  np.savetxt(os.path.join(file_path,datafile.split('.')[0]+".mean"), np.c_[train_stats['mean']])
  np.savetxt(os.path.join(file_path,datafile.split('.')[0]+".std"),  np.c_[train_stats['std']])
  np.savetxt(os.path.join(file_path,datafile.split('.')[0]+".max"),  np.c_[train_stats['max']])
  np.savetxt(os.path.join(file_path,datafile.split('.')[0]+".min"),  np.c_[train_stats['min']])
  # Save model predictions
  show_model_performance(os.path.join(file_path,datafile.split('.')[0]),model,normed_train_data,train_labels,normed_test_data,test_labels)
  # Clear the model
  clear_models()
  return None

def train(name='model',database=None,elem="H",ftra=0.8,vsplit=0.2,nepochs=100000,patnc=25,lr=0.000001,loss='mse',optimizer='RMSprop',neurons=(10,10,10),activations=('tanh','tanh','linear')):
  """
  Train atomistic FFNN models from a database stored (as a CSV list) in a Python variable
  name       : general name for the model files
  database   : database list in CSV format
  elem       : element for which the model will be trained
  tfra       : fraction of the total data to be used for training
  vsplit     : fraction used for the validation split
  nepochs    : number of training EPOCHS
  patience   : patience used to prevent over-fitting
  lr         : learning rate 
  loss       : statistical metric used as loss function
  optimizer  : tf.keras.optimizers to be used
  neurons    : list containing the shape of the hidden layers
  activations: list with the activations of each layer
  """
  file_path=os.getcwd() 
  # Check if the selected element is within the database
  if elem not in set([item[0] for item in database]): 
     print("Error: " + elem + " not in the database.")
     exit(0)
  # Check the optimizer
  if (optimizer not in tf.keras.optimizers.__dict__.keys()):
     print("Error: Unrecognized optimizer ", optimizer)
     print(tf.keras.optimizers.__dict__.keys())
     exit(0)
  else:
     opt_type=str('tf.keras.optimizers.') + str(optimizer)
  # Check the activation functions
  for i in activations:
      if i not in tf.keras.activations.__dict__.keys():
         print("Error: Unrecognized activation ", i)
         print(tf.keras.activations.__dict__.keys())
         exit(0)
  # Get data for the desired element:
  database = [s.strip() for s in database]
  filtr_database = [s.split(',')[1:] for s in database if s.startswith(elem)]
  inshp=(len(filtr_database[0])-1)
  rang    = range(1,inshp+1)
  colname = ['prop']
  for i in rang:
      colname.append('G' + str(i))
  # Get data from database
  raw_dataset = pd.DataFrame(filtr_database, columns =colname)
  raw_dataset = raw_dataset.apply(pd.to_numeric, errors='coerce')
  dataset = raw_dataset.copy()
  dataset = dataset.dropna()
  # Split and normalize the database
  train_dataset = dataset.sample(frac=ftra,random_state=0)
  test_dataset = dataset.drop(train_dataset.index)
  train_stats,train_labels=set_database(train_dataset)
  test_stats,test_labels=set_database(test_dataset)
  normed_train_data = norm(train_dataset,train_stats)
  normed_test_data = norm(test_dataset,train_stats)   
  normed_train_stats = normed_train_data.describe()
  normed_train_stats = normed_train_stats.transpose()
  normed_test_stats = normed_test_data.describe()
  normed_test_stats = normed_test_stats.transpose()
  # Set the model
  class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
      if epoch % 100 == 0: print('')
      print('.', end='')
  model = build_model(inshp,lr,loss,opt_type,neurons,activations)
  model.summary()
  # Train the model
  checkpoint_dir = os.path.dirname(file_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                   save_weights_only=True,
                                                   verbose=1)
  early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patnc)
  history=model.fit(normed_train_data, train_labels, epochs=nepochs,
                      validation_split =vsplit, verbose=1, callbacks=[early_stop, PrintDot(),cp_callback])
  # Save model
  model.save(os.path.join(file_path,name+"_"+elem+".h5"))
  # Save training statistics
  np.savetxt(os.path.join(file_path,name+"_"+elem+".mean"), np.c_[train_stats['mean']])
  np.savetxt(os.path.join(file_path,name+"_"+elem+".std"),  np.c_[train_stats['std']])
  np.savetxt(os.path.join(file_path,name+"_"+elem+".max"),  np.c_[train_stats['max']])
  np.savetxt(os.path.join(file_path,name+"_"+elem+".min"),  np.c_[train_stats['min']])
  # Save model predictions
  show_model_performance(os.path.join(file_path,name+"_"+elem),model,normed_train_data,train_labels,normed_test_data,test_labels)
  # Clear the model
  clear_models()
  return None

def xyz2dtbase(datafile="data.txt",itype="input.type",rtype="input.rad",atype="input.ang",fsave="yes"):
  """
  Return ACSF database (as a CSV) from a standard extended XYZ input file
  datafile: path to the database in extended xyz format
  itype   : path to the "input.type" file required for the computation of the ACSF features
  rtype   : path to the "input.rad" file required for the computation of the ACSF features
  atype   : path to the "input.ang" file required for the computation of the ACSF features
  save    : save database to an output file? (y/n)
  """
  file_path=(os.path.dirname(datafile))
  if len(file_path)==0 : file_path=os.path.join(os.getcwd()) 
  # Read input.type data
  with open(os.path.join(itype), 'r') as f:
      telem = int(f.readline().strip().split()[0])
      tipo = [f.readline().strip() for _ in range(telem)]
  # Read input.rad data
  with open(os.path.join(rtype), "r") as f:
      type_rad = int(f.readline().strip().split()[0])
      rcut_rad = float(f.readline().strip().split()[0])
      radmax   = int(f.readline().strip().split()[0])
      if type_rad == 1:        # Normal Radial ACSF
          rs_rad = np.zeros((telem, telem, radmax), dtype=np.float64)
          eta_rad = np.zeros((telem, telem, radmax), dtype=np.float64)
          nrad = np.zeros((telem, telem), dtype=np.int32)
          for i in range(1, telem**2 + 1):
              ni, nj, idum = map(int, f.readline().strip().split()[:3])
              nrad[ni-1, nj-1] = idum
              for k in range(1, idum + 1):
                  rs, eta = map(float, f.readline().strip().split()[:2])
                  rs_rad[ni-1, nj-1, k-1] = rs
                  eta_rad[ni-1, nj-1, k-1] = eta
      elif type_rad == 2:      # Z-Weighted Radial ACSF
          rs_rad = np.zeros((telem, 1, radmax), dtype=np.float64)
          eta_rad = np.zeros((telem, 1, radmax), dtype=np.float64)
          nrad = np.zeros((telem, 1), dtype=np.int32)
          for i in range(1, telem + 1):
              ni, idum = map(int, f.readline().strip().split()[:2])
              nrad[ni-1, 0] = idum
              for k in range(1, idum + 1):
                  rs, eta = map(float, f.readline().strip().split()[:2])
                  rs_rad[ni-1, 0, k-1] = rs
                  eta_rad[ni-1, 0, k-1] = eta
      else:
          raise ValueError("Unrecognizable Radial Symmetry Function Type")
  # Read input.ang data
  indjk = np.zeros(telem, dtype=int)
  vecino = np.zeros((telem*(telem+1)//2,), dtype=int)
  for i in range(1,telem+1):
      indjk[i-1] = (i-1)*(2*telem-i)
  counter = 0
  for i in range(telem):
      for j in range(i, telem):
          counter += 1
          vecino[counter-1] = indjk[i] + j + 1
  with open(os.path.join(atype), "r") as f:
       type_ang = int(f.readline().strip().split()[0])
       rcut_ang = float(f.readline().strip().split()[0])
       angmax = int(f.readline().strip().split()[0])
       idum = (telem*(telem+1)//2)
       if type_ang == 1:    # Normal
          rs_ang = np.zeros((telem, idum, angmax))
          xi_ang = np.zeros((telem, idum, angmax))
          eta_ang = np.zeros((telem, idum, angmax))
          lambda_ang = np.zeros((telem, idum, angmax))
          theta_s = np.zeros((telem, idum, angmax))
          nang = np.zeros((telem, idum),dtype=int)
          for i in range(1, telem*(telem*(telem+1)//2) + 1):
              ni, nj, idum = map(int, f.readline().strip().split()[:3])
              pepe = SFC.intpos(vecino, telem*(telem+1)//2, nj)-1
              nang[ni-1, pepe] = idum
              for k in range(1, idum + 1):
                  rs_ang[ni-1, pepe, k-1], xi_ang[ni-1, pepe, k-1], eta_ang[ni-1, pepe, k-1], lambda_ang[ni-1, pepe, k-1] = map(float, f.readline().strip().split()[:4])
       elif type_ang == 2:  # Modified
          rs_ang = np.zeros((telem, idum, angmax))
          xi_ang = np.zeros((telem, idum, angmax))
          eta_ang = np.zeros((telem, idum, angmax))
          lambda_ang = np.zeros((telem, idum, angmax))
          theta_s = np.zeros((telem, idum, angmax))
          nang = np.zeros((telem, idum),dtype=int)
          for i in range(1, telem * (telem * (telem + 1) // 2) + 1):
              ni, nj, idum = map(int, f.readline().strip().split()[:3])
              pepe = SFC.intpos(vecino, telem * (telem + 1) // 2, nj)-1
              nang[ni-1, pepe] = idum
              for k in range(1, idum + 1):
                  rs_ang[ni-1, pepe, k-1], xi_ang[ni-1, pepe, k-1], eta_ang[ni-1, pepe, k-1], lambda_ang[ni-1, pepe, k-1] = map(float, f.readline().strip().split()[:4])
       elif type_ang == 3:  # Heavily Modified
          rs_ang = np.zeros((telem, idum, angmax))
          xi_ang = np.zeros((telem, idum, angmax))
          eta_ang = np.zeros((telem, idum, angmax))
          theta_s = np.zeros((telem, idum, angmax))
          lambda_ang = np.zeros((telem, idum, angmax))
          nang = np.zeros((telem, idum),dtype=int)
          for i in range(1, telem * (telem * (telem + 1) // 2) + 1):
             ni, nj, idum = map(int, f.readline().strip().split()[:3])
             pepe = SFC.intpos(vecino, telem * (telem + 1) // 2, nj)-1
             nang[ni-1, pepe] = idum
             for k in range(1, idum + 1):
                 rs_ang[ni-1, pepe, k-1], xi_ang[ni-1, pepe, k-1], eta_ang[ni-1, pepe, k-1], theta_s[ni-1, pepe, k-1] = map(float, f.readline().strip().split()[:4])
       elif type_ang == 4:  # Z-Weighted
          idum = 1
          rs_ang = np.zeros((telem, idum, angmax))
          xi_ang = np.zeros((telem, idum, angmax))
          eta_ang = np.zeros((telem, idum, angmax))
          lambda_ang = np.zeros((telem, idum, angmax))
          theta_s = np.zeros((telem, idum, angmax))
          nang = np.zeros((telem, idum),dtype=int)
          for i in range(telem):
             ni,  idum = map(int, f.readline().strip().split()[:2])
             nang[ni-1, 0] = idum
             for k in range(1,idum+1):
                  rs_ang[ni-1, 0, k-1], xi_ang[ni-1, 0, k-1], eta_ang[ni-1, 0, k-1], lambda_ang[ni-1, 0, k-1] = map(float, f.readline().strip().split()[:4])
       else:
          raise ValueError('Unrecognizable Angular Symmetry Function Type')
  # Read database
  prop=[]
  acsf=[]
  element=[]
  with open(datafile,'r') as f:
       lines=f.readlines()
  count=0
  while count < len(lines):
       natoms=int(lines[count].split()[0].strip())
       label=[]
       coord=[]
       count +=2
       for i in np.arange(natoms):
           line = lines[count].split()
           label.append(line[0])
           element.append(line[0])
           coord.append(line[1:4])
           prop.append(line[4])
           count +=1
       coord=np.array(coord,float)
       # Compute ACSF functions for each geometry
       geom_acsf=(SFC.sfc_calc_for_database(label,coord,telem,tipo,type_rad,rcut_rad,radmax,nrad,rs_rad,eta_rad,indjk,vecino,type_ang,rcut_ang,angmax,rs_ang,xi_ang,eta_ang,lambda_ang,nang))
       [acsf.append(sublist) for sublist in geom_acsf]
  # Create the final database: it contains label, prop, acsf features.
  database=[]
  for i in np.arange(0,len(acsf)):
      database.append(str(element[i]) + "," + str(prop[i]) + ","+ ','.join(map(str,acsf[i])))
  # If required by the user, the database will be saved in output files
  if (fsave == "yes"):
      for i in set(element):
          outdatafile=os.path.join(file_path,datafile+"_"+i+".dtbse")
          with open(outdatafile,'w') as fout:
               # The element label is skipped in the final database files.
               fout.write('\n'.join([ele[2:] for ele in database if ele.split(',')[0] in [i]]))
  return database
