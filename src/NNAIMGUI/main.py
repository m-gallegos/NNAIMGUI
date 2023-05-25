"""
Main kernel of NNAIMGUI

Author: M. Gallegos, 2023.
"""

import numpy as np
import random
import argparse
import pandas as pd
from argparse import RawTextHelpFormatter
import sys
import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import pandas as pd
from random import randint
import random
from NNAIMGUI import SFC
from NNAIMGUI import dictionaries
from NNAIMGUI.config import *

def norm(x,mean,std):
   """
   Normalize the data.
   """
   y=np.empty_like(x)
   y[:]=x
   y=np.transpose(y)
   counter=0
   for i in y:
     y[counter,:]=(y[counter,:]-mean[counter])/(std[counter])
     counter=counter+1
   y=np.transpose(y)
   return y

def weight_calc():
   """
   Compute the weights used to correct the atomic charges in the charge 
   equilibration schemes.
   """
   w = []
   w.clear()
   if (args.charge_equilibration == 1):                  
      print(" # Charge equilibration ( ",args.charge_equilibration,") Homogeneous weight distribution")
      for i in np.arange(natoms):
          w.append(1.0/float(natoms))
   elif (args.charge_equilibration == 2):               
      print(" # Charge equilibration ( ",args.charge_equilibration,") Absolute charge weight distribution")
      for i in np.arange(natoms):
          w.append(abs(charges[i])/sum(abs(charges)))
   elif (args.charge_equilibration == 3):                
      print(" # Charge equilibration ( ",args.charge_equilibration,") Atomic populations weight distribution")
      for i in np.arange(natoms):
          w.append((Z[i]- charges[i])/sum(Z-charges))
   elif (args.charge_equilibration == 4):                
      print(" # Charge equilibration ( ",args.charge_equilibration,") Electronegativity (Sanderson) weight distribution")
      EN = []
      for i in np.arange(natoms):
          for key in dictionaries.sanderson_en:
              if (elements[i] == key):
                  EN.append(dictionaries.sanderson_en[key])
      EN = np.asarray(EN,dtype=float)
      for i in np.arange(natoms):
          w.append(EN[i]/sum(EN))
   elif (args.charge_equilibration == 5):                
      print(" # Charge equilibration ( ",args.charge_equilibration,") Electronegativity (Pauling) weight distribution")
      EN = []
      for i in np.arange(natoms):
          for key in dictionaries.pauling_en:
              if (elements[i] == key):
                  EN.append(dictionaries.pauling_en[key])
      EN = np.asarray(EN,dtype=float)
      for i in np.arange(natoms):
          w.append(EN[i]/sum(EN))
   elif (args.charge_equilibration == 6):                
      print(" # Charge equilibration ( ",args.charge_equilibration,") Sdev-error based weight distribution.")
      for i in np.arange(natoms):
          w.append(esdev[i]/sum(esdev))
   elif (args.charge_equilibration == 7):               
      print(" # Charge equilibration ( ",args.charge_equilibration,") Mean-error based weight distribution.")
      for i in np.arange(natoms):
          w.append(abs(emean[i])/sum(abs(emean)))
   elif (args.charge_equilibration == 8):              
      print(" # Charge equilibration ( ",args.charge_equilibration,") Combo 0")
      for i in np.arange(natoms):
          w.append(abs(emean[i])*esdev[i]/(sum(abs(emean)*esdev)))
   elif (args.charge_equilibration == 9):           
      print(" # Charge equilibration ( ",args.charge_equilibration,") Combo 1")
      for i in np.arange(natoms):
          w.append(abs(emean[i]/rango[i])*(Z[i]- charges[i])/sum(abs(emean)*(Z-charges)/rango))
   elif (args.charge_equilibration == 10):         
      print(" # Charge equilibration (",args.charge_equilibration,") Combo 2")
      for i in np.arange(natoms):
          w.append(abs(emean[i])*(Z[i]- charges[i])/sum(abs(emean)*(Z-charges)))
   elif (args.charge_equilibration == 11):       
      print(" # Charge equilibration (",args.charge_equilibration,") Combo 3")
      for i in np.arange(natoms):
          w.append((abs(emean[i])*esdev[i]*(Z[i]- charges[i])/(rango[i]))/sum(abs(emean)*esdev*(Z-charges)/(rango)))
   elif (args.charge_equilibration == 12):      
      print(" # Charge equilibration (",args.charge_equilibration,") Iterative Random Atomic Error Sampling")
      charges_iter = []
      for i in np.arange(natoms):
          charges_iter.append(charges[i])
      charges_iter=np.asarray(charges_iter,dtype=float)
      excs_q=sum(charges_iter)
      excs_q_zero=sum(charges_iter)
      contador=0
      while abs(excs_q) > abs(qtoler):
            contador +=1
            for i in np.arange(natoms):
                qcorr= (random.gauss(dictionaries.NNAIMQ_emean_val[elements[i]],dictionaries.NNAIMQ_esdev_val[elements[i]]/sigma))
                qcorr=qcorr*(abs(excs_q))*10
                charges_iter[i]=charges_iter[i]+qcorr*(dictionaries.atomicnumber[elements[i]]-charges[i])*(abs(dictionaries.NNAIMQ_emean_val[elements[i]]))
            excs_q=sum(charges_iter)
            if (excs_q*excs_q_zero < 0):
               print(" # Iterative procedure did not go as expected, the results were starting to diverge!")
               break
      print(" # Convergence achieved after ", contador, " iterations.")
      for i in np.arange(natoms):
            corr = charges[i]-charges_iter[i]
            w.append(corr/((sum(charges))*abs(charges[i])))
   elif (args.charge_equilibration == 13):     
      print(" # Charge equilibration (",args.charge_equilibration,") Iterative Random Elemental Error Sampling")
      charges_iter = []
      for i in np.arange(natoms):
          charges_iter.append(charges[i])
      charges_iter=np.asarray(charges_iter,dtype=float)
      excs_q=sum(charges_iter)
      excs_q_zero=sum(charges_iter)
      contador=0
      while abs(excs_q) > abs(qtoler):
            contador +=1
            qcorr = {}
            qcorr.clear()
            for key in chemspace:
                qcorr[key] = (random.gauss(dictionaries.NNAIMQ_emean_val[key],dictionaries.NNAIMQ_esdev_val[key]/sigma))*abs(excs_q)*10
            for i in np.arange(natoms):
                charges_iter[i]=charges_iter[i]+qcorr[elements[i]]*(dictionaries.atomicnumber[elements[i]]-charges[i])*(abs(dictionaries.NNAIMQ_emean_val[elements[i]]))
            excs_q=sum(charges_iter)
            if (excs_q*excs_q_zero < 0):
               print(" # Iterative procedure did not go as expected, the results were starting to diverge!")
               break
      print(" # Convergence achieved after ", contador, " iterations.")
      for i in np.arange(natoms):
            corr = charges[i]-charges_iter[i]
            w.append(corr/((sum(charges))*abs(charges[i])))
   w=np.asarray(w,dtype=float)
   return w

def charge_corr():
   """
   Computing the corrected atomic charges after the charge equilibration
   """
   charges_eq=[]
   charges_eq.clear()
   deltaq = sum(charges)
   for i in np.arange(natoms):
         corr = w[i]*deltaq*abs(charges[i])/sum(w*abs(charges))
         charges_eq.append(charges[i]-corr)
   charges_eq=np.asarray(charges_eq,dtype=float)
   return charges_eq

def print_prop(prop):
   """
   Print the atomic properties to the standard output
   """
   contador = 0
   print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
   print("                       Atomic Properties              ")
   print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
   print("        Atom Number " + "   Atom Label " + "   Atomic Properties " + "\n")
   for i in elements:
         print( "             " + "{0:3d}".format(contador+1) + "             "  + i + "         " + str("{: .4f}".format(prop[contador]) ))
         contador +=1
   print(" ")
   print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
   print(" # Reconstructed molecular property= ", "{:.3f}".format(sum(prop)))
   print(" ")
   return None

def save_file(nombre):
   """
   Save the atomic properties to an output file.
   """
   file_output=nombre+".nnaim"
   contador=0
   with open(os.path.join(output_dir,file_output), 'w') as fout:
        fout.write("        Atom Number " + "   Atom Label " + "   Atomic Property " + "\n")
        for i in elements:
           fout.write( "             " + "{0:3d}".format(contador+1) + "             "  + i + "         " + str("{: .4f}".format(prop[contador])+"\n" ))
           contador+=1
        fout.write("\n")
        fout.write(" # Reconstructed molecular property= "+ "{:.3f}".format(sum(prop)))
        fout.write("\n")
        fout.write(" # Target property      = "+ target_prop+"\n")
        fout.write(" # Units                = "+ prop_units+"\n")
        fout.write(" # Model used           = "+ model_folder+"\n")
        if (target_prop == "QTAIM charges"): fout.write(" # Charge equilibration = "+ str(ceq_num)+"\n")
   print(" # Data saved to file " + str(file_output))
   return None

def run_models():
   """
   Run the FFNN predictions.
   """
   geom=args.filename
   f=open(os.path.join(output_dir, geom), "r")
   contents=f.read()
   f.close()
   size=len(geom)
   nombre=geom[:size-4]
   # Launching TensorFlow
   np.set_printoptions(threshold=sys.maxsize)
   pd.set_option("display.max_rows", None, "display.max_columns", None)
   # Compute the ACSF features:
   acsf=SFC.sfc_calc(elements,coords,model_folder)
   # Run FFNN predictions:
   prop=np.empty(natoms,dtype=float)
   for etype in chemspace:
       df=[]
       atomnum=[]
       for i in np.arange(0,natoms):
           if elements[i] == str(etype):
              df.append(acsf[i])
              atomnum.append(i+1)
       atomnum=np.array(atomnum)
       if (len(df) != 0):
          df=np.array(df)
          raw_df = pd.DataFrame(df)
          data= raw_df.copy()
          data_stats=data.describe()
          data_stats=data_stats.transpose()
          normed_data=norm(data,globals()[f"mean_{etype}"],globals()[f"std_{etype}"])
          predictions=globals()[f"model_{str(etype)}"].predict(normed_data).flatten()
          for i in np.arange(0,len(atomnum)):
              prop[atomnum[i]-1]=predictions[i]
   return nombre,prop

# Set command line flags
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("-f", "--filename", help="\n"
                                             "Specify the name of the file containing the Cartesian coordinates (XYZ) in angstroms.\n"
                                             "\n",type=str)
parser.add_argument("-gui", "--gui_interface", choices=['yes', 'no'], default='no',help="\n"
                                                                                        "Do you want to launch the Graphical User Interface (GUI)?,\n"
                                                                                        "default= no\n"
                                                                                        "\n",type=str)
parser.add_argument("-fsave", "--fsave", choices=['yes', 'no'], default='yes',help="\n"
                                                                                        "Do you want to save the charges to an output file?,\n"
                                                                                        "default= yes\n"
                                                                                        "\n",type=str)
parser.add_argument("-ceq", "--charge_equilibration", choices=[-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13],default=0,help="\n"
                                                                                         "Specify the charge equilibration scheme used to refine the final atomic charges.\n"
                                                                                         "\n"
                                                                                         "++ -1   : Custom charge equilibration.\n"
                                                                                         "++  0   : Without charge equilibration (default).\n"
                                                                                         "++  1   : Homogeneous weight distribution.\n"
                                                                                         "++  2   : Charge based weight distribution.\n"
                                                                                         "++  3   : Atomic populations based weight distribution.\n"
                                                                                         "++  4   : Electronegativity (Sanderson) based weight distribution.\n"
                                                                                         "++  5   : Electronegativity (Pauling) based weight distribution.\n"
                                                                                         "++  6   : Sdev-error based weight distribution.\n"
                                                                                         "++  7   : Mean-error based weight distribution.\n"
                                                                                         "++  8   : Combo 0.\n"
                                                                                         "++  9   : Combo 1.\n"
                                                                                         "++  10  : Combo 2.\n"
                                                                                         "++  11  : Combo 3.\n"
                                                                                         "++  12  : Iterative Random Atomic Error Sampling.\n"
                                                                                         "++  13  : Iterative Random Elemental Error Sampling.\n"
                                                                                         "\n"
                                                                                         "Schemes 6-13 can only be used in combination with NNAIMQ models.\n"
                                                                                         "\n",type=int)
parser.add_argument("-ceq_file", "--charge_equilibration_file", default=initial_dir,help="\n"
                                             "Specify the path to the custom charge equilibration module.\n"
                                             "\n",type=str)
parser.add_argument("-sigma", "--sigma", default=1,help="\n"
                                             "Specify the number of sigmas used to bias the error distribution in the iterative charge equilibration.\n"
                                             "default= 1\n"
                                             "\n",type=float)
parser.add_argument("-qtoler", "--qtoler", default=1E-3,help="\n"
                                             "Specify the tolerance used in the iterative charge equilibration.\n"
                                             "default= 1E-3\n"
                                             "\n",type=float)
parser.add_argument("-model", "--model", choices=["NNAIMQ","Custom"],default="NNAIMQ",help="\n"
                                             "Specify the model to be used.\n"
                                             "++ NNAIMQ.\n"
                                             "++ Custom.\n"
                                             "\n",type=str)
parser.add_argument("-model_folder", "--model_folder",help="\n"
                                             "Specify the path to the model folder.\n"
                                             "\n",type=str)
parser.add_argument("-prop", "--prop", default="QTAIM charges",help="\n"
                                             "Specify the atomic property to be predicted.\n"
                                             "\n",type=str)
parser.add_argument("-units", "--units", default="electrons",help="\n"
                                             "Specify the units of the target atomic property.\n"
                                             "\n",type=str)
args = parser.parse_args()

if (args.gui_interface == "no"):
   # NO-GUI
   if getattr(sys, 'frozen', False):
       os.chdir(sys._MEIPASS)    # Changing to the temporary directory
   else:                         # Changing to the main location of the NNAIMGUI code
       os.chdir(pathname)
   print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
   print(" NNAIMGUI code.")
   print(" ")
   print(" M. Gallegos, 2023")
   print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
   print(" # No gui will be lunched")
   if (args.filename is None):
      print("\n")
      print(" # Error : no input file was given, aborting the run !")
      print("\n")
      parser.print_help()
      exit(0)
   else:
      filepath=(os.path.dirname(args.filename))
      if len(filepath)==0 : args.filename=os.path.join(initial_dir,args.filename)
      infile = open(args.filename, "r")
      output_dir=(os.path.dirname(args.filename))
      natoms = int(infile.readline())         
      natoms_check=0
      stuff= infile.readline()             
      elements = []
      elements.clear()
      coords = np.zeros((natoms,3),float)
      natoms_check =0
      for line in infile:
          label,x,y,z = line.split()
          elements.append(label)           
          coords[natoms_check,:] = x,y,z    
          natoms_check+=1
      infile.close()
      model_type=args.model
      if (model_type == "NNAIMQ"):
         model_folder=os.path.join(pathname,"models/NNAIMQ/")
         target_prop="QTAIM charges"
         prop_units="electrons"
      else:
         model_folder=args.model_folder
         target_prop=args.prop
         prop_units=args.units
      model_list=[]
      model_max=[]
      model_min=[]
      model_std=[]
      model_mean=[]
      chemspace=[]
      for file in os.listdir(model_folder):
          if file.endswith(".h5"):
              model_list.append(file)
              if (os.path.isfile(os.path.join(model_folder,file.strip().split(".")[0]+".max")))  : model_max.append(file.strip().split(".")[0]+".max")
              if (os.path.isfile(os.path.join(model_folder,file.strip().split(".")[0]+".min")))  : model_min.append(file.strip().split(".")[0]+".min")
              if (os.path.isfile(os.path.join(model_folder,file.strip().split(".")[0]+".std")))  : model_std.append(file.strip().split(".")[0]+".std")
              if (os.path.isfile(os.path.join(model_folder,file.strip().split(".")[0]+".mean"))) : model_mean.append(file.strip().split(".")[0]+".mean")
      if (len(model_list) == 0):
         print("\n")
         print(" # Error : no model files were found in folder.")
         print("\n")
         parser.print_help()
         exit(0)
      list_check=[model_list, model_max, model_min, model_std,model_mean]
      if not all(len(list_check[0]) == len(l) for l in list_check[1:]):
         print("\n")
         print(" # Error : model files seem to be incomplete! Check all .max, .min, .std, .mean and .h5 files in folder.")
         print("\n")
         parser.print_help()
         exit(0)
      for i in np.arange(0,len(model_list[0])):
          if (str(model_list[0])[i] != str(model_list[-1])[i]): model_rn=model_list[0][0:i]
      for i in np.arange(0,len(model_list)):
          chemspace.append(model_list[i].strip().split(".")[0][len(model_rn):])
          globals()[f"model_{str(chemspace[i])}"] = tf.keras.models.load_model(os.path.join(model_folder,model_list[i]))
          globals()[f"mean_{str(chemspace[i])}"] = np.loadtxt(os.path.join(model_folder,model_mean[i]),dtype='f')
          globals()[f"std_{str(chemspace[i])}"] = np.loadtxt(os.path.join(model_folder,model_std[i]),dtype='f')
          globals()[f"col_{str(chemspace[i])}"] = int(tf.keras.models.load_model(os.path.join(model_folder,model_list[i])).layers[0].input_shape[1])
      sigma=args.sigma
      qtoler=args.qtoler
      print(" # Input file         : "+ args.filename)
      print(" # Model type         : "+ model_type)
      print(" # Model folder       : "+ model_folder)
      print(" # Property           : "+ target_prop)
      print(" # Units              : "+ prop_units)
      print(" # Chemspace          : "+ ''.join(chemspace))
      print(" # Tensorflow version : "+ tf.version.VERSION)
      print(" # Sigma              : ", sigma)
      print(" # Qtoler             : ", qtoler)
      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      check = True
      for i in np.arange(natoms):
          if elements[i] not in chemspace:
             print(" # Error : element " +  str(i+1) + "("+elements[i]+") not in "+''.join(chemspace) + "\n")
             check = False
      if (check == True):          
         nombre,prop = run_models()
      else:
         print(" # Invalid XYZ input file :(")
         exit(0)
      if (target_prop == "QTAIM charges"):
         charges=prop.copy()
         Z     = []
         emean = []
         esdev = []
         mean  = []
         rango = []
         for i in np.arange(natoms):
             for key in chemspace: 
                 if (elements[i] == key):
                     Z.append(dictionaries.atomicnumber[key])
                     emean.append(dictionaries.NNAIMQ_emean_val[key])
                     esdev.append(dictionaries.NNAIMQ_esdev_val[key])
                     mean.append(dictionaries.NNAIMQ_mean_val[key])
                     rango.append(dictionaries.NNAIMQ_range_val[key])
         Z     = np.asarray(Z,dtype=float)
         emean = np.asarray(emean,dtype=float)
         esdev = np.asarray(esdev,dtype=float)
         mean  = np.asarray(mean,dtype=float)
         rango = np.asarray(rango,dtype=float)
         ceq_num= int(args.charge_equilibration)
         if (model_type != "NNAIMQ" and ceq_num in np.arange(6,14)):
            print(" # Error: Charge equilibration schemes 6-13 can only be used in combination with NNAIMQ models.")
            exit(0)
         if (ceq_num == 0):                  
            charges_eq = charges
         elif (ceq_num == -1):              
            ceq_file=args.charge_equilibration_file
            ceqpath=(os.path.dirname(ceq_file))
            if len(ceqpath)==0 : ceqpath=initial_dir
            sys.path.append(ceqpath)
            ceq_file=os.path.basename(ceq_file)
            ceq_file=ceq_file.split('.py')[0]
            custom_ceq = __import__(ceq_file)
            w= custom_ceq.weight_calc(charges,elements)
            charges_eq = charge_corr()
            prop=charges_eq.copy()
         else:                              
            w= weight_calc(ceq_num)
            charges_eq = charge_corr() 
            prop=charges_eq.copy()
      if (args.fsave == "yes"): 
          save_file(nombre)
      elif (args.fsave == "no"):
          print_prop(prop)
   print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
   print("   Normal termination of the code.")
   print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
elif (args.gui_interface == "yes"):
   # GUI
   print(" # Launching GUI interface.")
   from NNAIMGUI import gui
