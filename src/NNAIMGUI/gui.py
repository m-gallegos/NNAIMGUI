"""
M. Gallegos, 2023.

The GUI of the NNAIMGUI code. 
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from matplotlib.figure import Figure
import sys
import subprocess
import os
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

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)    # Change to the temporary directory
else:                         # Change to the main location of the code
    os.chdir(pathname)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
bond_cut = 0.95 
ref_height=1080 
ref_width=1920  
main_font="Helvetica"       
dialog_ttl="Information"    
chemspace=["C","H","O","N"] 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def set_axes_equal(ax):
    """
    A function to force the axis to exhibit an equal aspect ratio in the 
    3D visualization.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return None

def popup_info(title,info,duration):
    """
    Show popup information to the user
    """
    root2 = tk.Tk()
    root2.withdraw()
    root2.option_add('*Font', main_font + " " + str(main_fsize))
    root2.option_add('*Dialog.msg.font', main_font + " " + str(main_fsize))
    try:
       root2.after(duration, root2.destroy)
       tk.messagebox.showinfo(title=title, message=info,parent=root2)
    except:
       pass
    return None

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

def weight_calc(ceq_num):
   """
   Compute the weights used to correct the atomic charges in the charge 
   equilibration schemes. The weight coefficients will be stored as an array.
   """
   w=[]
   w.clear()
   if (ceq_num == 1):                
      for i in np.arange(natoms):
          w.append(1.0/float(natoms))
   elif (ceq_num == 2):              
      for i in np.arange(natoms):
          w.append(abs(charges[i])/sum(abs(charges)))
   elif (ceq_num == 3):             
      for i in np.arange(natoms):
          w.append((Z[i]- charges[i])/sum(Z-charges))
   elif (ceq_num == 4):            
      EN = []
      for i in np.arange(natoms):
          for key in dictionaries.sanderson_en:
              if (elements[i] == key):
                  EN.append(dictionaries.sanderson_en[key])
      EN = np.asarray(EN,dtype=float)
      for i in np.arange(natoms):
          w.append(EN[i]/sum(EN))
   elif (ceq_num == 5):           
      EN = []
      for i in np.arange(natoms):
          for key in dictionaries.pauling_en:
              if (elements[i] == key):
                  EN.append(dictionaries.pauling_en[key])
      EN = np.asarray(EN,dtype=float)
      for i in np.arange(natoms):
          w.append(EN[i]/sum(EN))
   elif (ceq_num == 6):          
      for i in np.arange(natoms):
          w.append(esdev[i]/sum(esdev))
   elif (ceq_num == 7):         
      for i in np.arange(natoms):
          w.append(abs(emean[i])/sum(abs(emean)))
   elif (ceq_num == 8):        
      for i in np.arange(natoms):
          w.append(abs(emean[i])*esdev[i]/(sum(abs(emean)*esdev)))
   elif (ceq_num == 9):               
      for i in np.arange(natoms):
          w.append(abs(emean[i]/rango[i])*(Z[i]- charges[i])/sum(abs(emean)*(Z-charges)/rango))
   elif (ceq_num == 10):             
      for i in np.arange(natoms):
          w.append(abs(emean[i])*(Z[i]- charges[i])/sum(abs(emean)*(Z-charges)))
   elif (ceq_num == 11):               
      for i in np.arange(natoms):
          w.append((abs(emean[i])*esdev[i]*(Z[i]- charges[i])/(rango[i]))/sum(abs(emean)*esdev*(Z-charges)/(rango)))
   elif (ceq_num == 12):         
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
               info_box.insert(tk.END, "\n")
               info_box.insert(tk.END," # Iterative procedure did not go as expected, the results were starting to diverge!" + "\n")
               tk.messagebox.showinfo(title=dialog_ttl, message="Iterative equilibration failed!",parent=root)
               break
      info_box.insert(tk.END, "\n")
      info_box.insert(tk.END," # Convergence achieved after "+ str(contador) + " iterations." + "\n")
      for i in np.arange(natoms):
            corr = charges[i]-charges_iter[i]
            w.append(corr/((sum(charges))*abs(charges[i])))
   elif (ceq_num == 13):                 
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
               info_box.insert(tk.END, "\n")
               info_box.insert(tk.END," # Iterative procedure did not go as expected, the results were starting to diverge!" + "\n")
               tk.messagebox.showinfo(title=dialog_ttl, message="Iterative equilibration failed!",parent=root)
               break
      info_box.insert(tk.END, "\n")
      info_box.insert(tk.END," # Convergence achieved after "+ str(contador) + " iterations." + "\n")
      for i in np.arange(natoms):
            corr = charges[i]-charges_iter[i]
            w.append(corr/((sum(charges))*abs(charges[i])))
   w=np.asarray(w,dtype=float)
   return w

def charge_corr():
   """
   Compute the corrected atomic charges after the charge equilibration
   """
   charges_eq=[]
   charges_eq.clear()
   deltaq = sum(charges)
   for i in np.arange(natoms):
         corr = w[i]*deltaq*abs(charges[i])/sum(w*abs(charges))
         charges_eq.append(charges[i]-corr)
   charges_eq=np.asarray(charges_eq,dtype=float)
   return charges_eq

def get_xyz():
   """
   Function to select the xyz file and extract its info
   """
   global elements, coords, natoms, xyz_file, output_dir

   run_button.configure(state='disabled')
   visualize_button.configure(state='disabled')
   save_button.configure(state='disabled')
   picture_button.configure(state='disabled')
   info_box.configure(state='normal') 
   xyz_file = filedialog.askopenfilename(title="Select XYZ file",initialdir = initial_dir)
   output_dir=(os.path.dirname(xyz_file))
   info_box.insert(tk.END,"\n"+"---------------------------------------------------------------------"+ "\n")
   info_box.insert(tk.END,"Starting new calculation")
   info_box.insert(tk.END,"\n"+"---------------------------------------------------------------------"+ "\n")
   info_box.insert(tk.END,"\n"+"(1) Will read info from " + str(xyz_file) + "\n")
   info_box.insert(tk.END,"\n"+"Printing file contents : " + "\n")
   infile=open(xyz_file,'r')
   natoms = int(infile.readline())
   stuff= infile.readline() # Skip random line
   info_box.insert(tk.END,"\n"+str(natoms)+ "\n")
   info_box.insert(tk.END,"\n"+stuff+ "\n")
   elements=[]
   elements.clear()
   coords = np.zeros((natoms,3),float)
   natoms_check =0
   for line in infile:
       info_box.insert(tk.END,line)
       label,x,y,z = line.split()
       elements.append(label)             
       coords[natoms_check,:] = x,y,z     
       natoms_check+=1
   infile.close()
   info_box.insert(tk.END,"\n" + "\n")
   info_box.insert(tk.END,"\n"+"---------------------------------------------------------------------"+ "\n")
   run_button.configure(state='normal')
   info_box.insert(tk.END,"Ready to launch the FFNN models, please remember to set the desired charge equilibration scheme, Sigma and Qtoler values.")
   info_box.insert(tk.END,"\n" + "---------------------------------------------------------------------"+ "\n")
   info_box.see(tk.END)   
   info_box.configure(state='disabled') 
   return None

def save_file():
   """
   Save the atomic charges to the output file.
   """
   global ceq_num, nombre, charges_eq, prop

   info_box.configure(state='normal')
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
        fout.write(" # Units                = "+ prop_unit+"\n")
        fout.write(" # Model used           = "+ model_folder+"\n")
        if (target_prop == "QTAIM charges"): fout.write(" # Charge equilibration = "+ str(ceq_num)+"\n")
   info_box.insert(tk.END,"\n" + " # Data saved to file" + str(file_output)+ "\n")
   info_box.insert(tk.END, "\n")
   info_box.see(tk.END)   
   info_box.configure(state='disabled')
   popup_info(dialog_ttl,"Data saved to "+str(file_output),500)
   return None

def save_picture():
   """
   Render the plot of the atomic charges
   """
   global pic_name
   def render_image():
       # Retrieve picture information
       pic_size_x = int(pic_size_x_entry.get())
       pic_size_y = int(pic_size_y_entry.get())
       pic_dpi    = int(pic_dpi_entry.get())
       pic_cbar = var_pic_cbar_list.get()
       pic_title  = str(pic_title_entry.get())
       pic_setcbar_min = pic_setcbar_min_entry.get()
       pic_setcbar_max = pic_setcbar_max_entry.get()
       pic_setor_azim = pic_setor_azim_entry.get()
       pic_setor_elv  = pic_setor_elv_entry.get()
       pic_setcbar = [pic_setcbar_min,pic_setcbar_max]
       pic_setor   = [pic_setor_elv,pic_setor_azim]
       format_label= format_label_combobox.get()
   
       atom_size=[]
       atom_size.clear()
       color = []
       color.clear()
       p_labels = []
       p_labels.clear()
       for i in np.arange(natoms):
           atom_size.append(dictionaries.radii[elements[i]])
           if (vis_color == 'Element'): color.append(dictionaries.celem[elements[i]])
           if (vis_color == 'Atomic property'): color.append(prop[i])
           if (vis_label == 'Element'): p_labels.append(elements[i]+str(i+1))
           if (vis_label == 'Atomic property'): p_labels.append(format_label.format(prop[i]))
       atom_size=np.asarray(atom_size,dtype=float)
       atom_size=atom_size*25*(wp*hp)/(ref_width*ref_height)
       figure = plt.figure(figsize=(pic_size_x,pic_size_y),dpi=pic_dpi) 
       ax_pic = figure.add_subplot(projection='3d')
       for i in np.arange(len(coords[:,0])-1):                
          for j in np.arange(i+1,len(coords[:,0])):           
           atomi = np.array((coords[i,0],coords[i,1],coords[i,2])) 
           atomj = np.array((coords[j,0],coords[j,1],coords[j,2])) 
           dist = np.linalg.norm(atomi-atomj)          
           if (dist <= bond_cut*(0.5*(dictionaries.radii[elements[i]]+dictionaries.radii[elements[j]]))):
              ax_pic.plot3D([atomi[0], atomj[0]], [atomi[1], atomj[1]], [atomi[2], atomj[2]], color = "black")
       ax_pic.scatter(xs=coords[:,0],ys=coords[:,1],zs=coords[:,2],s=atom_size,c=color,edgecolor='black',cmap=vis_map,alpha=1)
       if (pic_title != None): plt.title(pic_title)
       if (pic_cbar == 'yes'):  
          clb = figure.colorbar(ax_pic.scatter(xs=coords[:,0],ys=coords[:,1],zs=coords[:,2],s=atom_size,c=color,edgecolor='black',cmap=vis_map,alpha=1), ax=ax_pic, shrink=0.4, pad = 0.1)
          clb.ax.set_title(prop_unit,fontsize=12*int((wp*hp)/(ref_width*ref_height)))
          if (pic_setcbar != None):
             clb.mappable.set_clim(vmin=pic_setcbar[0], vmax=pic_setcbar[1])
             clb.draw_all()
       ax_pic.grid(False)
       ax_pic.set_axis_off()
       if (vis_label != 'None'):
          for i, txt in enumerate(p_labels):
              ax_pic.text(coords[i,0]+0.05, coords[i,1]+0.05,coords[i,2]+0.05,txt).set_clip_on(True)
       ax_pic.view_init(float(pic_setor[0]),float(pic_setor[1])) # Adjust plot orientation
       set_axes_equal(ax_pic)
       plt.savefig(os.path.join(output_dir,pic_name),dpi=pic_dpi,transparent=True,bbox_inches='tight')
       tk.messagebox.showinfo(title=dialog_ttl, message="The file has been saved!",parent=picture_dialog)
       plt.close()
       picture_dialog.destroy()
   
   pic_name = tk.simpledialog.askstring(title=dialog_ttl,prompt="Select the output file",parent=root)
   if (pic_name.split('.')[-1] != 'png'):
       tk.messagebox.showinfo(title="Error", message="Output file must have the png extension!")
       return None
   else:
       picture_dialog=tk.Toplevel(root)
       picture_dialog_w = wp/3
       picture_dialog_h = hp/3
       picture_dialog_x = (wp/2) - (picture_dialog_w/2)
       picture_dialog_y = (hp/2) - (picture_dialog_h/2)
       picture_dialog.geometry('%dx%d+%d+%d' % (picture_dialog_w, picture_dialog_h, picture_dialog_x, picture_dialog_y))
       picture_dialog.title(dialog_ttl)
       local_font = (main_font, main_fsize)
       picture_dialog.option_add('*Font', main_font + " " + str(main_fsize))
       picture_dialog.option_add('*Dialog.msg.font', main_font + " " + str(main_fsize))
       rows=15
       columns=4
       for i in np.arange(rows):
           picture_dialog.grid_rowconfigure(i, weight=1)
       for i in np.arange(columns):
           picture_dialog.grid_columnconfigure(i, weight=1)
       vis_map= vis_map_combobox.get()
       tk.Label(picture_dialog, text="Size (in inches)", font=local_font).grid(row=1, column=0, padx=0, pady=0,sticky="nsew")
       pic_size_x_entry = tk.Entry(picture_dialog, textvariable=tk.StringVar(picture_dialog, value="6"), font=local_font)
       pic_size_x_entry.grid(row=1, column=1, padx=0, pady=0,sticky="nsew")
       pic_size_y_entry = tk.Entry(picture_dialog, textvariable=tk.StringVar(picture_dialog, value="8"), font=local_font)
       pic_size_y_entry.grid(row=1, column=2, padx=0, pady=0,sticky="nsew")
       tk.Label(picture_dialog, text="Resolution (dpi)", font=local_font).grid(row=3, column=0, padx=0, pady=0,sticky="nsew")
       pic_dpi_entry = tk.Entry(picture_dialog, textvariable=tk.StringVar(picture_dialog, value="350"), font=local_font)
       pic_dpi_entry.grid(row=3, column=1, padx=0, pady=0,sticky="nsew")
       tk.Label(picture_dialog, text="Show color bar", font=local_font).grid(row=5, column=0, padx=0, pady=0,sticky="nsew")
       pic_cbar_list = [
       "yes",
       "no",
       ]
       var_pic_cbar_list = tk.StringVar(picture_dialog)
       var_pic_cbar_list.set(pic_cbar_list[0])
       var_pic_cbar_opt = tk.OptionMenu(picture_dialog, var_pic_cbar_list, *pic_cbar_list)
       var_pic_cbar_opt.config(font=local_font)
       var_pic_cbar_opt.grid(row=5, column=1, padx=0, pady=0,sticky="nsew")
       pic_cbar = var_pic_cbar_list.get()
       def callback_cbar(*args):
           global pic_cbar  
           pic_cbar = var_pic_cbar_list.get()
       var_pic_cbar_list.trace("w", callback_cbar)
       tk.Label(picture_dialog, text="Title", font=local_font).grid(row=7, column=0, padx=0, pady=0,sticky="nsew")
       pic_title_entry= tk.Entry(picture_dialog,textvariable=tk.StringVar(picture_dialog,value=" "),font=local_font)
       pic_title_entry.grid(row=7, column=1, padx=0, pady=0,sticky="nsew")
       tk.Label(picture_dialog, text="Limit", font=local_font).grid(row=9, column=0, padx=0, pady=0,sticky="nsew")
       pic_setcbar_min_entry = tk.Entry(picture_dialog, textvariable=tk.StringVar(picture_dialog,value=str("{0:.2f}".format(min(prop)))), font=local_font)
       pic_setcbar_min_entry.grid(row=9, column=1, padx=0, pady=0,sticky="nsew")
       pic_setcbar_max_entry = tk.Entry(picture_dialog, textvariable=tk.StringVar(picture_dialog,value=str("{0:.2f}".format(max(prop)))), font=local_font)
       pic_setcbar_max_entry.grid(row=9, column=2, padx=0, pady=0,sticky="nsew")
       current_elv=(ax_main_plot.elev)
       current_azim=(ax_main_plot.azim)
       tk.Label(picture_dialog,text="Elevation (º)",font=local_font).grid(row=9,column=0,padx=0,pady=0,sticky="nsew")
       pic_setor_elv_entry= tk.Entry(picture_dialog,textvariable=tk.StringVar(picture_dialog,value=str("{0:.1f}".format(float(current_elv)))),font=local_font)
       pic_setor_elv_entry.grid(row=9,column=1,pady=0,sticky="nsew")
       tk.Label(picture_dialog,text="Azimuthal (º)",font=local_font).grid(row=11,column=0,padx=0,pady=0,sticky="nsew")
       pic_setor_azim_entry= tk.Entry(picture_dialog,textvariable=tk.StringVar(picture_dialog,value=str("{0:.1f}".format(float(current_azim)))),font=local_font)
       pic_setor_azim_entry.grid(row=11,column=1,pady=0,sticky="nsew")
       render_button = tk.Button(picture_dialog, text="Render", font=local_font, command=render_image)
       render_button.grid(row=13, column=0, columnspan=4,padx=0, pady=0,sticky="nsew")
       return None
   return None

def visualize():
   """
   Function for the 3D visualization of the molecule and the atomic charges.
   """
   global plot_frame, ax_main_plot

   picture_button.configure(state='normal')
   plot_frame.destroy()
   plot_frame = tk.Frame(root, bg='white', bd=1.5)
   plot_frame.grid(row=4, column=4, padx=0, pady=0, columnspan=8, rowspan=14, sticky="nsew")
   atom_size=[]
   atom_size.clear()
   color = []
   color.clear()
   p_labels = []
   p_labels.clear()
   format_label= format_label_combobox.get()
   for i in np.arange(natoms):
       atom_size.append(dictionaries.radii[elements[i]])
       if (vis_color == 'Element'): color.append(dictionaries.celem[elements[i]])
       if (vis_color == 'Atomic property'): color.append(prop[i])
       if (vis_label == 'Element'): p_labels.append(elements[i]+str(i+1))
       if (vis_label == 'Atomic property'): p_labels.append(format_label.format(prop[i]))
   atom_size=np.asarray(atom_size,dtype=float)
   atom_size=atom_size*75*(wp*hp)/(ref_width*ref_height)
   info_box.configure(state='normal')
   info_box.insert(tk.END,"\n" + " # Launching visualization functionalities." + "\n")
   info_box.insert(tk.END, "\n")
   info_box.see(tk.END) 
   figure = plt.Figure(figsize=(5,6), dpi=120) 
   ax_main_plot = figure.add_subplot(111,projection='3d')
   line = FigureCanvasTkAgg(figure, plot_frame)
   line.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=True)
   line.figure.tight_layout()
   line.draw()
   vis_map= vis_map_combobox.get()
   for i in np.arange(len(coords[:,0])-1):                   
       for j in np.arange(i+1,len(coords[:,0])):             
           atomi = np.array((coords[i,0],coords[i,1],coords[i,2])) 
           atomj = np.array((coords[j,0],coords[j,1],coords[j,2])) 
           dist = np.linalg.norm(atomi-atomj)         
           if (dist <= bond_cut*(0.5*(dictionaries.radii[elements[i]]+dictionaries.radii[elements[j]]))):
              ax_main_plot.plot3D([atomi[0], atomj[0]], [atomi[1], atomj[1]], [atomi[2], atomj[2]], color = "black")
   ax_main_plot.scatter(xs=coords[:,0],ys=coords[:,1],zs=coords[:,2],s=atom_size,c=color,edgecolor='black',cmap=vis_map,alpha=1)
   if (vis_color == 'Atomic property'):  
      clb = figure.colorbar(ax_main_plot.scatter(xs=coords[:,0],ys=coords[:,1],zs=coords[:,2],s=atom_size,c=color,edgecolor='black',cmap=vis_map,alpha=1), ax=ax_main_plot, shrink=0.9, pad = 0.1)
      clb.ax.set_title(prop_unit,fontsize=12*int((wp*hp)/(ref_width*ref_height)))
   ax_main_plot.grid(False)
   ax_main_plot.set_axis_off()
   if (vis_label != 'None'):
      for i, txt in enumerate(p_labels):
          ax_main_plot.text(coords[i,0]+0.05, coords[i,1]+0.05,coords[i,2]+0.05,txt).set_clip_on(True)
   set_axes_equal(ax_main_plot)
   info_box.see(tk.END)
   info_box.configure(state='disabled')
   return None

def run_models():
   """
   Run the atomistic predictions.
   """
   global chemspace, target_prop, prop_unit, nombre, Z, esdev, emean, mean, rango, prop, charges, w, charges_eq, qtoler, sigma, ceq_num, nombre, model_folder
   info_box.configure(state='normal')
   visualize_button.configure(state='disabled')
   save_button.configure(state='disabled')
   picture_button.configure(state='disabled')
   geom=xyz_file
   size=len(geom)
   nombre=geom[:size-4]
   f=open(geom,"r")
   contents=f.read()
   f.close()
   try :
       sigma = float(sigma_entry.get())
       info_box.insert(tk.END,"\n" +" # Sigma set to " + str(sigma) + "\n")
   except ValueError:
       info_box.insert(tk.END,"\n" +"Error : Wrong input value in sigma! " + str(sigma_entry.get()) + "\n")
       run_button.configure(state='disabled')
   try:
       qtoler = float(qtoler_entry.get())
       info_box.insert(tk.END,"\n" +" # Qtoler set to " + str(qtoler) + "\n")
   except ValueError:
       info_box.insert(tk.END,"\n"+"Error : Wrong input value in Qtoler! " + str(qtoler_entry.get()) + "\n")
       run_button.configure(state='disabled')
   model_type = model_selection.get()
   np.set_printoptions(threshold=sys.maxsize)
   pd.set_option("display.max_rows", None, "display.max_columns", None)
   info_box.insert(tk.END,"\n" + " # Tensorflow version :"+ tf.version.VERSION + "\n")

   if (model_type == "Custom"):  # Custom model
      popup_info(dialog_ttl,"A customized model will be used ! ",500)
      model_folder = filedialog.askdirectory(title="Select the path to the model folder",initialdir = initial_dir)
      target_prop=tk.simpledialog.askstring(dialog_ttl,"Indicate the name of the target property")
      prop_unit=tk.simpledialog.askstring(dialog_ttl,"Indicate the units of the target property")
   elif (model_type == "NNAIMQ"): # General NNAIMQ model (to be included with the main code)
      model_folder = os.path.join(pathname,"models/NNAIMQ/")
      target_prop="QTAIM charges"
      prop_unit="electrons"

   targprop_label.configure(state='normal')
   targprop_label.delete(0,tk.END)
   targprop_label.insert(0,target_prop)
   targprop_label.configure(state='disabled')
   units_label.configure(state='normal')
   units_label.delete(0,tk.END)
   units_label.insert(0,prop_unit)
   units_label.configure(state='disabled')
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
      tk.messagebox.showinfo(title="Error",message="No model files were found in folder.")
      run_button.configure(state='disabled')
      info_box.insert(tk.END,"\n" + " # Error loading the models!"+"\n")
      info_box.see(tk.END)   
      info_box.configure(state='disabled')
      return None
   popup_info(dialog_ttl,str(len(model_list))+ " different FFNN models were found!",2000)
   list_check=[model_list, model_max, model_min, model_std,model_mean]
   if not all(len(list_check[0]) == len(l) for l in list_check[1:]):
      tk.messagebox.showinfo(title="Error",message="Model files seem to be incomplete! Check all .max, .min, .std, .mean and .h5 files in folder.")
      run_button.configure(state='disabled')
      info_box.insert(tk.END,"\n" + " # Error loading the models!"+"\n")
      info_box.see(tk.END)   
      info_box.configure(state='disabled')
      return None
   for i in np.arange(0,len(model_list[0])):
       if (str(model_list[0])[i] != str(model_list[-1])[i]): model_rn=model_list[0][0:i]
   for i in np.arange(0,len(model_list)):
       chemspace.append(model_list[i].strip().split(".")[0][len(model_rn):])
       info_box.insert(tk.END,"\n" + " # Loading data for FFNN "+str(chemspace[i])+" model" + "\n")
       globals()[f"model_{str(chemspace[i])}"] = tf.keras.models.load_model(os.path.join(model_folder,model_list[i]))
       globals()[f"mean_{str(chemspace[i])}"] = np.loadtxt(os.path.join(model_folder,model_mean[i]),dtype='f')
       globals()[f"std_{str(chemspace[i])}"] = np.loadtxt(os.path.join(model_folder,model_std[i]),dtype='f')
       globals()[f"col_{str(chemspace[i])}"] = int(tf.keras.models.load_model(os.path.join(model_folder,model_list[i])).layers[0].input_shape[1])
   popup_info(dialog_ttl,"Checking chemical composition of the system.",500)
   for i in np.arange(natoms):
       if elements[i] not in chemspace:
           tk.messagebox.showinfo(title="Error",message="Element "+str(i+1)+"("+elements[i]+") not in "+''.join(chemspace)+".")
           run_button.configure(state='disabled')
           info_box.insert(tk.END,"\n" + " # Error in input XYZ file."+"\n")
           info_box.see(tk.END)   
           info_box.configure(state='disabled')
           return None
   popup_info(dialog_ttl,"Looking for ACSF descriptor files.",500)
   if  (len([f for f in ["input.rad","input.ang","input.type"] if os.path.exists(os.path.join(model_folder,f))]) != 3):
      tk.messagebox.showinfo(title="Error",message="ACSF feature files are missing! Check all input.rad input.ang and input.type files in folder.")
      run_button.configure(state='disabled')
      info_box.insert(tk.END,"\n" + " # Error loading the ACSF feature files!"+"\n")
      info_box.see(tk.END)   
      info_box.configure(state='disabled')
      return None
   popup_info(dialog_ttl,"Computing ACSF features.",500)
   acsf=SFC.sfc_calc(elements,coords,model_folder)
   info_box.insert(tk.END,"\n" + " # ACSF features computed !"+"\n")

   info_box.insert(tk.END,"(2) Running FFNN predictions" + "\n")
   prop=np.empty(natoms,dtype=float)
   for etype in chemspace:
       info_box.insert(tk.END,"Running predictions for "+ etype+" atoms" + "\n")
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
      popup_info(dialog_ttl,"Equilibrating NNAIMQ charges",500)
      info_box.insert(tk.END,"Running charge equilibration" + "\n")
      ceq_num= int(ceq.split()[0])
      if (model_type != "NNAIMQ" and ceq_num in np.arange(6,14)):
         tk.messagebox.showinfo(title="Error",message="Charge equilibration schemes 6-13 can only be used in combination with NNAIMQ models!")
         run_button.configure(state='disabled')
         info_box.insert(tk.END,"\n" + " # Error during charge equilibration!"+"\n")
         info_box.see(tk.END)   
         info_box.configure(state='disabled')
         return None

      if (ceq_num == 0):                
         charges_eq = charges
      elif (ceq_num == -1):            
         ceq_file = filedialog.askopenfilename(title="Select the path to the custom charge equilibration file",initialdir = initial_dir)
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
   info_box.insert(tk.END,"\n" + "---------------------------------------------------------------------"+ "\n")
   info_box.insert(tk.END,"                         Atomic properties                 ")
   info_box.insert(tk.END,"\n" + "---------------------------------------------------------------------"+ "\n")
   for i in np.arange(natoms):
       info_box.insert(tk.END,"Atom" + "     " + "{0:3d}".format(i+1) + "    " + elements[i] + "   " + "{: .3f}".format(prop[i]) + "\n")
   info_box.insert(tk.END,"\n" + "---------------------------------------------------------------------"+ "\n")
   info_box.insert(tk.END,"Molecular sum = " + "{: .3f}".format(sum(prop)) + " "+prop_unit+"." +"\n")
   info_box.insert(tk.END,"\n" + "---------------------------------------------------------------------"+ "\n")
   info_box.see(tk.END)   
   info_box.configure(state='disabled')
   visualize_button.configure(state='normal')
   save_button.configure(state='normal')
   return None

def print_info():
   """
   Print some basic info about the NNAIMGUI code and its use.
   """
   infostr="""
This is a simple Graphical User Interface (GUI) of the NNAIMGUI code. The latter is designed for the prediction, equilibration and visualization of QTAIM atomic charges using Neural Network models. It comes with an already built model (NNAIMQ) for the estimation of QTAIM charges of C, H, O and N containing (gas-phase, neutral and singlet spin) molecules at the M06-2X/def2-TZVP level of theory. Additionally, customized models can be loaded to predict any other atomic property of choice.

The code requires a regular input file specifying the geometry of the system to be predicted (as given by a general XYZ file). The input file format must be the following:

    # natoms
    # empty line
    # label r(3)

Once loaded, the atomic properties are obtained after executing the "Run FFNN" button. 

Besides the NN prediction of atomic charges, NNAIMQ includes multiple charge equilibration schemes to ensure that the total neutral molecular character is reconstructed. Sigma defines the scaling factor of the standard deviation used in the equilibration schemes relying on random error sampling approaches. Similarly, Qtoler sets the threshold (in electrons) for the excess of molecular charge to be used in the iterative refinements of the local charges.

Once predicted, the atomic properties can be visualized with the "Visualize" button. This will create an interactive 3D representation of the system. The atoms can be colored according to their chemical identity or to the value of the atomic property (as given by the VisCl variable), using the coloring scheme set by "VisMap". If specified in "VisLbl", the actual value of the atomic properties can be displayed on the screen. 

The results can be saved to a standard output file (with ".nnaim" extension) or to a picture (in ".png" format) for further analysis.

For further information, check:

   # Miguel Gallegos, José Manuel Guevara-Vela, and Ángel Martín Pendás , "NNAIMQ: A neural network model for predicting QTAIM charges", J. Chem. Phys. 156, 014112 (2022)
   # https://github.com/m-gallegos/NNAIMQ.
   # https://github.com/m-gallegos/NNAIMGUI.
   """
   text_box.configure(state='normal')   
   text_box.insert(tk.END,infostr)
   text_box.configure(state='disabled')
   return None

def show_chemspace():
   infortmp="""
Available chemical diversity:
"""
   for i in chemspace:
        infortmp += str(i)
   tk.messagebox.showinfo(message=infortmp, title=dialog_ttl)
   return None
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Start main tkinter window 
root = tk.Tk()
root.title('NNAIMGUI')
wp, hp = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (wp, hp))
wp=float(wp)
hp=float(hp)
main_fsize = min(int(hp / 90), int(wp / 160))

# Check display size
ref_width=float(ref_width)
ref_height=float(ref_height)
real_ratio=wp/hp
desired_ratio=ref_width/ref_height
if (real_ratio != desired_ratio):
    popup_info(dialog_ttl,"Screen display does not have the 16:9 aspect ratio: expect a weird looking GUI.",5000)

# Set main configuration
root.option_add('*Font', main_font + " " + str(main_fsize))
root.option_add('*Dialog.msg.font', main_font + " " + str(main_fsize))
img = tk.PhotoImage(file="./data/logo.png")
root.tk.call('wm','iconphoto', root._w,img)
columns=12
for i in np.arange(columns):
    root.grid_columnconfigure(i, weight=1,minsize=100,uniform="column")
rows=18
for i in np.arange(rows):
    root.grid_rowconfigure(i, weight=1,minsize=52,uniform="row")

text_box = tk.Text(root, font=(main_font, main_fsize))
text_box.configure(state='disabled')
text_box.grid(row=0, column=0, columnspan=12, rowspan=2,padx=0, pady=0,sticky="news")
print_info()

tk.Label(root,text="Sigma",font=(main_font, main_fsize)).grid(row=2,column=0,padx=0,pady=0,sticky='news')
sigma_entry = tk.Entry(root,textvariable=tk.StringVar(root,value="1"),font=(main_font, main_fsize))
sigma_entry.grid(row=2,column=1,padx=0,pady=0,sticky='news')

tk.Label(root,text="Qtoler",font=(main_font, main_fsize)).grid(row=2,column=2,padx=0,pady=0,sticky='news')
qtoler_entry = tk.Entry(root,textvariable=tk.StringVar(root,value="1E-3"),font=(main_font, main_fsize))
qtoler_entry.grid(row=2,column=3,padx=0,pady=0,sticky='news')

CeqList = [
"-1   : Custom charge equilibration",
"0    : No charge equilibration (default)",
"1    : Homogeneous weight distribution",
"2    : Charge based weight distribution",
"3    : Atomic populations based weight distribution",
"4    : Electronegativity (Sanderson) based weight distribution",
"5    : Electronegativity (Pauling) based weight distribution",
"6    : Sdev-error based weight distribution",
"7    : Mean-error based weight distribution",
"8    : Combo 0",
"9    : Combo 1",
"10   : Combo 2",
"11   : Combo 3",
"12   : Iterative Random Atomic Error Sampling (IRAES)",
"13   : Iterative Random Elemental Error Sampling (IREES)"
]
variable = tk.StringVar(root)
variable.set(CeqList[1])
opt = tk.OptionMenu(root, variable, *CeqList)
opt.config(font=(main_font, main_fsize))
opt.grid(row=3, column=0, padx=0, pady=0, rowspan=1,columnspan=4,sticky="nsew")
ceq = variable.get()
def callback(*args):
    global ceq 
    ceq = variable.get()
variable.trace("w", callback)

open_button = tk.Button(root, text="Open XYZ file", font=(main_font, main_fsize), command=get_xyz)
open_button.grid(row=4, column=0, padx=0, pady=0, columnspan=4,sticky="nsew")

run_button = tk.Button(root, text="Run FFNN", font=(main_font, main_fsize), command=run_models)
run_button.grid(row=5, column=0, padx=0, pady=0, columnspan=4,sticky="nsew")
run_button.configure(state='disabled')

visualize_button = tk.Button(root, text="Visualize", font=(main_font, main_fsize), command=visualize)
visualize_button.grid(row=6, column=0, padx=0, pady=0, columnspan=4,sticky="nsew")
visualize_button.configure(state='disabled')

save_button = tk.Button(root, text="Save file", font=(main_font, main_fsize), command=save_file)
save_button.grid(row=7, column=0, padx=0, pady=0, columnspan=4,sticky="nsew")
save_button.configure(state='disabled')

picture_button = tk.Button(root, text="Save picture", font=(main_font, main_fsize), command=save_picture)
picture_button.grid(row=8, column=0, padx=0, pady=0,columnspan=4, sticky="nsew")
picture_button.configure(state='disabled')

info_box = tk.Text(root, font=(main_font, main_fsize))
info_box.configure(state='disabled')
info_box.grid(row=9, column=0, padx=0, pady=0, rowspan=9,columnspan=4,sticky="news")

vis_color_list = [
"Element",
"Atomic property"
]
var_color_list = tk.StringVar(root)
var_color_list.set(vis_color_list[0])
var_color_opt = tk.OptionMenu(root, var_color_list, *vis_color_list)
var_color_opt.config(font=(main_font, main_fsize))
tk.Label(root,text="VisCl",font=(main_font, main_fsize)).grid(row=2, column=4, padx=0, pady=0, sticky="news")
var_color_opt.grid(row=3, column=4, padx=0, pady=0, sticky="news")
vis_color = var_color_list.get()
def callback_color(*args):
    global vis_color  
    vis_color = var_color_list.get()
var_color_list.trace("w", callback_color)
vis_label_list = [
"None",
"Element",
"Atomic property"
]
var_label_list = tk.StringVar(root)
var_label_list.set(vis_label_list[0])
var_label_opt = tk.OptionMenu(root, var_label_list, *vis_label_list)
var_label_opt.config(font=(main_font, main_fsize))
tk.Label(root,text="VisLbl",font=(main_font, main_fsize)).grid(row=2, column=5, padx=0, pady=0, sticky="news")
var_label_opt.grid(row=3, column=5, padx=0, pady=0, sticky="news")
vis_label = var_label_list.get()
def callback_label(*args):
    global vis_label  
    vis_label = var_label_list.get()
var_label_list.trace("w", callback_label)
tk.Label(root,text="VisMap",font=(main_font, main_fsize)).grid(row=2, column=6, padx=0, pady=0, sticky="news")
vis_map_list= []
for mapa in plt.colormaps():
    vis_map_list.append(mapa)
vis_map = tk.StringVar()
vis_map.set("rainbow_r")
vis_map_combobox = ttk.Combobox(root, textvariable=vis_map, font=(main_font, main_fsize),values=vis_map_list)
vis_map_combobox['state'] = 'readonly'
vis_map_combobox.grid(row=2, column=7, padx=0, pady=0, sticky="news")

tk.Label(root,text="Label format",font=(main_font, main_fsize)).grid(row=3,column=6,padx=0,pady=0,sticky='news')
format_label_list = ["{: .3f}","{: .2f}","{: .1f}"]
format_label = tk.StringVar()
format_label.set(format_label_list[0])
format_label_combobox = ttk.Combobox(root, textvariable=format_label, font=(main_font, main_fsize),values=format_label_list)
format_label_combobox['state'] = 'readonly'
format_label_combobox.grid(row=3,column=7,padx=0,pady=0,sticky='news')

tk.Label(root,text="Model: ",font=(main_font, main_fsize)).grid(row=2, column=8, padx=0, pady=0, sticky="news")
model_list= ["NNAIMQ", "Custom"]
model = tk.StringVar()
model.set(model_list[0])
model_selection = ttk.Combobox(root, textvariable=model, font=(main_font, main_fsize),values=model_list)
model_selection['state'] = 'readonly'
model_selection.grid(row=2, column=9, padx=0, pady=0, sticky="news")

tk.Label(root,text="Target prop: ",font=(main_font, main_fsize)).grid(row=3, column=8, padx=0, pady=0, sticky="news")
targprop_label=tk.Entry(root,textvariable=tk.StringVar(root,value="QTAIM charges"),font=(main_font, main_fsize),state='disabled')
targprop_label.grid(row=3, column=9, padx=0, pady=0, sticky="news")
targprop_label.configure(disabledbackground="white",disabledforeground="black")

chemspace_button = tk.Button(root,text="Check Chem Space",font=(main_font, main_fsize),command=show_chemspace)
chemspace_button.grid(row=2, column=10, columnspan=2,padx=0, pady=0, sticky="news")

tk.Label(root,text="Units: ",font=(main_font, main_fsize)).grid(row=3,column=10,pady=0,padx=0,sticky='news')
units_label=tk.Entry(root,textvariable=tk.StringVar(root,value="electrons"),font=(main_font, main_fsize),state='disabled')
units_label.grid(row=3,column=11,padx=0,pady=0,sticky='news')
units_label.configure(disabledbackground="white",disabledforeground="black")

plot_frame = tk.Frame(root, bg='white', bd=1.5)
plot_frame.grid(row=4, column=4, padx=0, pady=0, columnspan=8, rowspan=14, sticky="nsew")

default_font = tk.font.nametofont("TkDefaultFont")
default_font.configure(size=main_fsize)
root.option_add("*Font", default_font)
tk.mainloop() 
