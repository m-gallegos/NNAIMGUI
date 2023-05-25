import numpy as np
from NNAIMGUI.dictionaries import *

# A custom module for the equilibration of the 
# atomic charges relying on the relative size
# of the atoms.

# The module must contain the weight_calc subruoutine 
# which receives as input the charges and the element list and
# returns an array with the corresponding weights for each atom.

def weight_calc(charges,elements):
   w = []
   w.clear()
   natoms=int(len(elements))
   size=[]
   for i in np.arange(natoms):
       size.append(radii[elements[i]])
   tot_size=sum(size)
   for i in np.arange(natoms):
       w.append(size[i]/tot_size)
   w=np.asarray(w,dtype=float)
   return  w
