# NNAIMGUI

https://doi.org/10.1021/acs.jcim.3c00597

v.0.0.1
M. Gallegos, University of Oviedo, 2023

NNAIMGUI is a code for the prediction and visualization of atomic properties, specifically oriented for the estimation of QTAIM quantities,**[1]** such as atomic charges, using Feed Forward Neural Network (FFNN) models. The code comes with a built-in model, NNAIMQ **[2]**, for the prediction of Bader partial charges of gas-phase, neutral and singlet-spin molecules comprising C, H, O and N atoms. However, the user is free to load customized models trained to predict any atomic property of interest. As a proof of concept, we have included a rough model trained on a 10% subsample of the NNAIMQ database capable of computing the localized electron populations of CHON containing systems (can be found in **/src/examples/models/LIAIM/**). Furthermore, the code implements a module for building atomistic models which combines a built-in featurization approach to allow non-experienced users to construct FFNNs in a simple and effective way.

If atomic charges are to be predicted, NNAIMGUI accounts for a charge equilibration approach to ensure the electroneutral character of the resultant molecule (the build-up of local errors in the estimation of the atomic properties may result in the non-ideal reconstruction of the net molecular charge). The code accounts for a total of 13 different algorithms for the distribution of the atomic corrections, while tailor-made equilibration strategies can be easily loaded.

The code can be executed from the command line or interactively using the intuitive and easy-to-use Graphical User Interface (GUI) running under Python, while being compatible with both Linux and Windows operating systems.

# Installation

NNAIMGUI can be easily installed using the pip Python package manager:

    pip install git+https://github.com/m-gallegos/NNAIMGUI.git

Alternatively, one can download the zip file from the NNAIMGUI GitHub and run the following command:

    pip install NNAIMGUI-main.zip

The code requires other Python modules to run which will be installed along with the former (if not present in the local machine). To avoid incompatibilities, it is generally advisable to install and run NNAIMGUI under a Python environment:

    python -m venv venv
    source /venv/bin/activate  (Linux) or venv\Scripts\activate      (Windows)

**Note: if you encounter problems while installing NNAIMGUI do not hesitate to contact us. We will provide support as well as the source or built distributions of the code (if needed).**

# Command line execution

In order to execute NNAIMGUI directly from the command line, use the given script (/src/main.py) with the following flag arguments:

    -f filename     : name of the xyz file of the molecule (.xyz extension).
    -gui (yes/no)   : launch the GUI?.
    -fsave (yes/no) : save predictions to an output file?. 
    -model          : model to be used (NNAIMQ/Custom).

If a Custom model is selected, the following flags should also be specified:

    -model_folder   : path to the model folder (only required for Custom models).
    -prop           : name of the target property (only required for Custom models).
    -units          : units of the target property (only required for Custom models).

If charge equilibration schemes are to be applied to the predicted QTAIM charges, the following flags should be specified.

    -ceq            : charge equilibration scheme used to refine the atomic charges (-1-13). (optional)
    -ceq_file       : path to the custom charge equilibration module, with a .py extension. (optional)
    -sigma          : number of sigmas used to bias the error distribution in the iterative charge equilibration. (optional)
    -qtoler         : threshold value for the residual molecular charge in the iterative charge equilibration. (optional)

Additional information and details about the command-line execution flags of NNAIMGUI can be obtained with the following command:

    python main.py --help

# GUI execution

The GUI of the code can be invoked from the previous **main.py** script:

    python main.py -gui yes

Or directly from a Python terminal by importing the **gui** module of **NNAIMGUI**:

    >>> from NNAIMGUI import gui

After running any of the previous commands, the GUI of the code should pop-up on the screen. NNAIMGUI has been designed for an ideal resolution of 1920:1080, aspect ratios differing from the latter may result in a weird looking interface. Any time a change in the visualization parameters is performed, the molecular representation should be updated by re-clicking on the **Visualize** button.

# Notes

* Output files are stored with a **.nnaim** extension in the same folder where the starting geometry file was located.
* Charge equilibration is only available if the target property is **QTAIM charges**. 
* Charge equilibration schemes 6-13 rely on model specific parameter and thus can only be used in combination with the built-in NNAIMQ model.
* Although in future versions of the code different input formats may be accepted, NNAIMGUI currently requires the geometries to be provided in standard XYZ Cartesian Coordinates (in Angstroms). However, converting other chemical structure formats to XYZ can be readily done with Open Babel **[3]**, for instance:

#
    obabel geom.pdb -O geom.xyz 

# Custom FFNN models:

Custom FFNN models can be used to predict any target atomic property of choice. NNAIMGUI has a built-in molecular featurization module (SCF) that allows to convert XYZ Cartesian coordinates into Atom Centered Symmetry Function (ACSF) **[4]** features which are then fed into the NN models. The following block of code shows how to load a FFNN model trained to predict the QTAIM localization index (can be found at /examples/LIAIM).

    python main.py -f geom.xyz -gui no -fsave no -model Custom -model_folder ./examples/LIAIM -prop "Localization index" -units "electrons"

On the other hand, if NNAIMGUI is run interactively, after clicking the "Run FFNN" button the code will ask the user to provide information
about the model folder along with the property name and units.

The model folder must contain two different types of files: 

* ACSF  files: **input.type**, **input.rad** and **input.ang**. These files contain the information required to compute the ACSF features.

**input.type**: in this file, the id of each element type in the ACSF feature space must be specified. These id numbers will be then used to set the element-specific contributions to the ACSF environment of a given atom. Let's imagine, for instance, a model for water clusters, where the chemical diversity is limited to H and O atoms:

    2   # Number of different element types
    H   # H will be element type 1
    O   # O will be element type 2

**input.rad**: in this file, the type and parameters of the radial contributions of each neighboring element to the ACSF features of a given atom must be specified. Two different radial symmetry function kernels are currently implemented in NNAIMGUI:

Normal Radial Symmetry Function, **[4,5]** as given by:
```math
G^{rad}_{i} = \sum^{N}_{j \ne i} e^{-\eta(r_{ij}-r_{s})^{2}} \cdot fc(r_{ij}).
```
Atomic Weighted Radial Symmetry Functions, **[6]** as given by:
```math
W^{rad}_{i} = \sum^{N}_{j \ne i} g(Z_j) \cdot e^{-\eta(r_{ij}-r_{s})^{2}} \cdot fc(r_{ij}),
```


    1      # Type of radial sym function (1 normal, 2 Z-weighted)
    10.0   # Rcut in Angstroms
    5      # Maximum number of radial functions for a given pair
    1 1 5  # (atom type, neighboring atom type, number of functions) followed by the Rs and Eta values of each function
    0.0000000000000000     0.50000000000000000      
    0.0000000000000000     0.13224489795918368       
    0.0000000000000000     5.9911242603550297E-002  
    0.0000000000000000     3.4026465028355393E-002  
    0.0000000000000000     2.1903731746890212E-002 
    1 2 5  # O environment for H
    0.0000000000000000     0.50000000000000000      
    0.0000000000000000     0.13224489795918368      
    0.0000000000000000     5.9911242603550297E-002 
    0.0000000000000000     3.4026465028355393E-002 
    0.0000000000000000     2.1903731746890212E-002 
    2 1 5  # H environment for O
    0.0000000000000000     0.50000000000000000     
    0.0000000000000000     0.13224489795918368     
    0.0000000000000000     5.9911242603550297E-002
    0.0000000000000000     3.4026465028355393E-002
    0.0000000000000000     2.1903731746890212E-002
    2 2 5  # O environment for O
    0.0000000000000000     0.50000000000000000     
    0.0000000000000000     0.13224489795918368     
    0.0000000000000000     5.9911242603550297E-002
    0.0000000000000000     3.4026465028355393E-002
    0.0000000000000000     2.1903731746890212E-002
    
The file will contain as many radial function blocks as possible atomic pairs, in our case 4: (1,1), (1,2) (2,1), (2,2). Notice that (1,2) and (2,1) are not equivalent as the former refers to the radial contribution of neighboring O atoms to the ACSF features of H atoms whereas the latter is the contribution of neighboring H atoms to a given O atom in the molecule.

**input.ang**:in this file, the type and parameters of the angular contributions of each neighboring pair to the ACSF features of a given atom must be specified. Currently, NNAIMGUI accounts for a total of 4 different angular symmetry function kernels:

Normal Angular Symmetry function, **[4,5]** as given by:
```math      
                G^{ang}_{i} = 2^{1-\xi}\sum^{N}_{j \ne i} \sum^{N}_{k \ne i,j} [(1 + \lambda \cdot cos \theta_{ijk})^{\xi}
                \cdot e^{-\eta [(r_{ij}-r_s)^2 + (r_{ik}-r_s)^2 +  (r_{jk}-r_s)^2]}\cdot f_c(r_{ij}) \cdot f_c(r_{ik}) \cdot f_c(r_{jk})].
```     

Modified Angular Symmetry function, **[4]** as given by:
 ```math       
                G^{ang}_{i} = 2^{1-\xi}\sum^{N}_{j \ne i} \sum^{N}_{k \ne i,j} [(1 + \lambda \cdot cos \theta_{ijk})^{\xi}
                \cdot e^{-\eta [(r_{ij}-r_s)^2 + (r_{ik}-r_s)^2] }\cdot f_c(r_{ij}) \cdot f_c(r_{ik})].
```        
Heavily Modified Angular Symmetry Function, **[7]** as given by:
```math
G^{ang}_{i} = 2^{1-\xi} \sum^{N}_{j,k \ne i} (1 + cos(\theta_{ijk}-\theta_{s}))^{\xi}
\cdot exp \left [ -\eta \left ( \frac{r_{ij} + r_{ik}}{2} -r_s\right )^2 \right ] \cdot f_c(r_{ij}) \cdot f_c(r_{ik}).
```
Atomic Weighted Angular Symmetry Functions, **[6]** as given by:
```math
                        W^{ang}_{i} = 2^{1-\xi} \cdot h(Z_j,Z_k) \cdot \sum^{N}_{j \ne i} \sum^{N}_{k \ne i,j} [(1 + \lambda \cdot cos \theta_{ijk})^{\xi}
                        \cdot e^{-\eta [(r_{ij}-r_s)^2 + (r_{ik}-r_s)^2 +  (r_{jk}-r_s)^2]}\cdot f_c(r_{ij}) \cdot f_c(r_{ik}) \cdot f_c(r_{jk})],
```

    3      # Angular sym function type (1 normal, 2 modified, 3 heavily modified, 4 Z-weighted)
    10.0   # Rcut in Angstroms
    5      # Maximum number of angular functions for a given pair
    1 1 5  # (atom, neighboring pair, number of functions) followed by the Rs, Xi, Eta, Lambda/Theta values
    0.0000000000000000        1.0000000000000000        1.0000000000000002E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        1.9036541768875849E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        3.6238984729102551E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        6.8986482119689760E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        0.13132640382981661       0.0000000000000000   
    1 2 5 # HO/OH environment of H atoms
    0.0000000000000000        1.0000000000000000        1.0000000000000002E-002   0.0000000000000000    
    0.0000000000000000        1.0000000000000000        1.9036541768875849E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        3.6238984729102551E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        6.8986482119689760E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        0.13132640382981661       0.0000000000000000   
    1 4 5 # OO environment of H atoms
    0.0000000000000000        1.0000000000000000        1.0000000000000002E-002   0.0000000000000000    
    0.0000000000000000        1.0000000000000000        1.9036541768875849E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        3.6238984729102551E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        6.8986482119689760E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        0.13132640382981661       0.0000000000000000   
    2 1 5 # HH environment of O atoms
    0.0000000000000000        1.0000000000000000        1.0000000000000002E-002   0.0000000000000000    
    0.0000000000000000        1.0000000000000000        1.9036541768875849E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        3.6238984729102551E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        6.8986482119689760E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        0.13132640382981661       0.0000000000000000   
    2 2 5 # HO/OH environment of O atoms
    0.0000000000000000        1.0000000000000000        1.0000000000000002E-002   0.0000000000000000    
    0.0000000000000000        1.0000000000000000        1.9036541768875849E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        3.6238984729102551E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        6.8986482119689760E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        0.13132640382981661       0.0000000000000000   
    2 4 5 # OO environment of O atoms
    0.0000000000000000        1.0000000000000000        1.0000000000000002E-002   0.0000000000000000    
    0.0000000000000000        1.0000000000000000        1.9036541768875849E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        3.6238984729102551E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        6.8986482119689760E-002   0.0000000000000000   
    0.0000000000000000        1.0000000000000000        0.13132640382981661       0.0000000000000000   

It should be noticed that now, the second id number in the heading of each block does not refer to an atom type but to an atomic pair type. These ids are obtained following a very simple recipe detailed in the Supporting information of the original NNAIMGUI reference, however the user can quickly check these id numbers with the aid of the following built-in function, which receives as input the number of element types:

    >>> from NNAIMGUI import SFC
    >>> SFC.show_neighmat(2)
    Element i 1 and Element j 1, id : 1
    Element i 1 and Element j 2, id : 2
    Element i 2 and Element j 2, id : 4
    
The input.ang will contain then as many angular function blocks as possible atomic trios, formed from the combination of a given atom type and all the possible different neighboring atomic pair. Notice that now, the heteroatomic neighboring pairs are equivalent owing to the kernel used to compute the ACSF functions (e.g H+O or O+H neighboring pairs have the same id (2)).

For further information about the functional form of the radial and angular symmetry functions included in the SFC module, and their constituting parameters, check the NNAIMGUI reference.

* model files: nn **(name)** X.h5, nn **(name)** X.std, nn **(name)** X.mean, nn **(name)** X.min and nn **(name)** X.max. Where X is the chemical symbol of the element for which each atomistic model has been trained for and **name** is any name of your choice. The .h5 file is the actual Tensorflow model in an h5 format. The .std, .mean, .min and .max are plain text files containing the standard deviation, mean, minimum and maximum value taken by each ACSF feature througout the training, which will be used to homogeneously standardize the data. The model folder must contain as many sets of model files (.h5, .std, .mean, .min, .max) as element specific atomistic models are present. For further details, check the example given in the **/src/examples/LIAM/** path. 

**Note:** The available chemical diversity will be determined based on the elements (X) appearing in each sets of model files, so the use of standard chemical symbols is mandatory.

# Custom Charge equilibration

If desired, the user is free to use a tailor-made charge equilibration, different from those provided by default with the NNAIMGUI code. The latter will require the user to specify the path to the corresponding Python file (with **.py** extension) which must contain a function (**weight_calc**) which receives as input the charges and the list of chemical symbols (stored in the **elements** list) and returns the weights in an array (**w**). As an example, the following block of code shows a charge equilibration scheme where the correction is made proportional to the Van der Waals radii of the atoms:

    import numpy as np
    from NNAIMGUI.dictionaries import *

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

After computing the weights, the charges are corrected as follows:

 ```math
q^{pred*}_A = q^{pred}_A - \frac{w_A \cdot \Delta Q \cdot |q^{pred}_A|}{\sum_{A=1,N}w_A \cdot |q^{pred}_A|}.
 ```

# Training models in NNAIMGUI

As an additional feature, NNAIMGUI implements a built-in module (trainer) for building and training new atomistic models. In this way the module allows non-experienced users to train simple FFNN models in a simple and effective way, which can be later used to run predictions on NNAIMGUI. 

The first step is transforming the standard XYZ coordinates into a ML database comprising the target property and the ACSF features of each atom. This can be done with the aid of the xyz2dtbase built-in function:
    
    >>> from NNAIMGUI import trainer
    >>> database=trainer.xyz2dtbase(datafile="test.xyz",itype="input.type",rtype="input.rad",atype="input.ang",fsave='yes')

Where datafile is the database in extended XYZ format and itype, rtype and atype are the input.type, input.rad and input.ang files gathering the main parameters of the chemical featurization. If the fsave option is set to "yes", element-specific database files will be saved in the local directory. The extended XYZ file employs the following format:

    natom                   # number of atoms of geom 1
                            # comment line
    label x y z prop        # Chemical symbol, R(:) coordinates, atomic property of a given atom
    . . . .
    . . . .
    natom                   # number of atoms of geom 2
                            # comment line
    label x y z prop 
    . . . .
    . . . .
    
Once the database has been created, the models can be trained from the previously stored files (if saved), as:

    >>> trainer.train_from_csv(datafile="test.xyz_H.dtbse",ftra=0.8,vsplit=0.2,nepochs=100000,patnc=25,lr=0.000001,loss='mse',optimizer='RMSprop',neurons=(10,10,10),activations=('tanh','tanh','linear'))

Taking as parameters:
    
    ftra       : fraction of data used for training
    vsplit     : validation split factor
    nepochs    : maximum number of EPOCHS allowed for training
    patnc      : patience of the early-stopping approach
    lr         : learning rate
    loss       : loss metric used to track the progress of the training
    optimizer  : model optimizer
    neurons    : list containing the number of neurons of the hidden layers
    activations: list containing the activation functions of the model
    
Or alternatively, by parsing the database directly to the train function while specifying the element for which the model will be trained:

    >>>trainer.train(name="test",database=database,elem="C",ftra=0.8,vsplit=0.2,nepochs=100000,patnc=25,lr=0.000001,loss='mse',optimizer='RMSprop',neurons=(10,10,10),activations=('tanh','tanh','linear'))

In this way, all possible atomisitc models for the chemical diversity of the database can be easily built in a few lines of code:
    
    from NNAIMGUI import trainer
    database=trainer.xyz2dtbase(datafile="test.xyz",itype="input.type",rtype="input.rad",atype="input.ang",fsave='yes')
    for elem in set([item[0] for item in database]):
        trainer.train(database=database,elem=elem,ftra=0.8,vsplit=0.2,nepochs=100,patnc=25,lr=0.000001,loss='mse',optimizer='RMSprop',neurons=(10,10,10),activations=('tanh','tanh','linear'))

**Note:** for the sake of simplicity, the current NNAIMGUI trainer module uses deeply connected FFNN models.

# References
**[1]** R. Bader, Atoms in Molecules: A Quantum Theory, of International series of monographs on chemistry, Oxford University Press, Oxford, 1990.

**[2]** (a) M. Gallegos, J.M. Guevara-Vela, and A. Martín Pendás , The Journal of Chemical Physics, 156, 014112 (2022). (b)  https://github.com/m-gallegos/NNAIMQ

**[3]** O'Boyle, N.M., Banck, M., James, C.A. et al. The Journal of Cheminformatics, 3, 33 (2011). 

**[4]** J. Behler , The Journal of Chemical Physics, 134, 074106 (2011).

**[5]** J. Behler and M. Parrinello, Physical Review Letters, 98, 146401 (2007).

**[6]** M. Gastegger, L. Schwiedrzik, M. Bittermann, F. Berzsenyi and P. Marquetand, The Journal of Chemical Physics, 148, 241709 (2018).

**[7]** J. S. Smith, O. Isayev and A. E. Roitberg, Chemical Science, 8, 3192–3203 (2017).
