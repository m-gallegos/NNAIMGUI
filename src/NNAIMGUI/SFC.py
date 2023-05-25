"""
ACSF calculator module

Author: M. Gallegos, 2023

This module contains some basic functions for the
computation of ACSF features used in FFNN predictions.
"""

import numpy as np
import sys as sys
import os
from NNAIMGUI import dictionaries

def show_neighmat(telem):
    ind   = np.zeros(telem, dtype=int)
    for i in range(1,telem+1):
        ind[i-1] = (i-1)*(2*telem-i)
    for i in range(telem):
        for j in range(i, telem):
            print("Element i "+str(i+1)+" and Element j "+str(j+1)+", id : "+str(ind[i] + j + 1))
    return None

def intpos(vec, nele, val):
    for i in range(nele):
        if vec[i] == val:
            return i + 1
    return 1

def chemclas(eti, telem, tipo):
    clas = 0
    for i in range(telem):
        if eti == tipo[i]:
            clas = i + 1
            break
    if clas == 0:
        print(f"Fatal Error: {eti} not recognized as a known element.")
        raise SystemExit("Error Termination")
    return clas

def distance(atomi, atomj):
    dist = 0.0
    for i in range(3):
        dist += (atomj[i] - atomi[i])**2
    dist = np.sqrt(dist)
    return dist

def cutoff(cut, dist):
    pi = np.pi
    if dist > cut:
        fcut = 0.0
    elif dist <= cut:
        fcut = 0.5 * (np.cos(pi * dist / cut) + 1.0)
    return fcut

def dotproduct(vec1, vec2):
    dot = np.dot(vec1, vec2)
    return dot

def atomfactor(Z):
    factor = Z
    return factor

def pairfactor(Zi, Zj):
    factor = Zi * Zj
    return factor

def sfc_calc(label,coord,model_path):
    # Read element types
    with open(os.path.join(model_path,'input.type'), 'r') as f:
        telem = int(f.readline().strip().split()[0]) 
        tipo = [f.readline().strip() for _ in range(telem)] 
    # Read radial ACSF
    with open(os.path.join(model_path,"input.rad"), "r") as f:
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
    # Read Angular ACSF 
    indjk = np.zeros(telem, dtype=int)
    vecino = np.zeros((telem*(telem+1)//2,), dtype=int)
    for i in range(1,telem+1):
        indjk[i-1] = (i-1)*(2*telem-i)
    counter = 0
    for i in range(telem):
        for j in range(i, telem):
            counter += 1
            vecino[counter-1] = indjk[i] + j + 1
    with open(os.path.join(model_path,"input.ang"), "r") as f:
         type_ang = int(f.readline().strip().split()[0])  
         rcut_ang = float(f.readline().strip().split()[0]) 
         angmax = int(f.readline().strip().split()[0]) 
         idum = (telem*(telem+1)//2)
         if type_ang == 1:    # Normal
            rs_ang = np.zeros((telem, idum, angmax))
            xi_ang = np.zeros((telem, idum, angmax))
            eta_ang = np.zeros((telem, idum, angmax))
            lambda_ang = np.zeros((telem, idum, angmax))
            nang = np.zeros((telem, idum),dtype=int)
            for i in range(1, telem*(telem*(telem+1)//2) + 1):
                ni, nj, idum = map(int, f.readline().strip().split()[:3]) 
                pepe = intpos(vecino, telem*(telem+1)//2, nj)-1 
                nang[ni-1, pepe] = idum 
                for k in range(1, idum + 1):
                    rs_ang[ni-1, pepe, k-1], xi_ang[ni-1, pepe, k-1], eta_ang[ni-1, pepe, k-1], lambda_ang[ni-1, pepe, k-1] = map(float, f.readline().strip().split()[:4]) 
         elif type_ang == 2:  # Modified
            rs_ang = np.zeros((telem, idum, angmax))
            xi_ang = np.zeros((telem, idum, angmax))
            eta_ang = np.zeros((telem, idum, angmax))
            lambda_ang = np.zeros((telem, idum, angmax))
            nang = np.zeros((telem, idum),dtype=int)
            for i in range(1, telem * (telem * (telem + 1) // 2) + 1):
                ni, nj, idum = map(int, f.readline().strip().split()[:3])
                pepe = intpos(vecino, telem * (telem + 1) // 2, nj)-1
                nang[ni-1, pepe] = idum
                for k in range(1, idum + 1):
                    rs_ang[ni-1, pepe, k-1], xi_ang[ni-1, pepe, k-1], eta_ang[ni-1, pepe, k-1], lambda_ang[ni-1, pepe, k-1] = map(float, f.readline().strip().split()[:4])
         elif type_ang == 3:  # Heavily Modified
             rs_ang = np.zeros((telem, idum, angmax))
             xi_ang = np.zeros((telem, idum, angmax))
             eta_ang = np.zeros((telem, idum, angmax))
             theta_s = np.zeros((telem, idum, angmax))
             nang = np.zeros((telem, idum),dtype=int)
             for i in range(1, telem * (telem * (telem + 1) // 2) + 1):
                ni, nj, idum = map(int, f.readline().strip().split()[:3])
                pepe = intpos(vecino, telem * (telem + 1) // 2, nj)-1
                nang[ni-1, pepe] = idum
                for k in range(1, idum + 1):
                    rs_ang[ni-1, pepe, k-1], xi_ang[ni-1, pepe, k-1], eta_ang[ni-1, pepe, k-1], theta_s[ni-1, pepe, k-1] = map(float, f.readline().strip().split()[:4])
         elif type_ang == 4:  # Z-Weighted
             idum = 1
             rs_ang = np.zeros((telem, idum, angmax))
             xi_ang = np.zeros((telem, idum, angmax))
             eta_ang = np.zeros((telem, idum, angmax))
             lambda_ang = np.zeros((telem, idum, angmax))
             nang = np.zeros((telem, idum),dtype=int)
             for i in range(telem):
                ni,  idum = map(int, f.readline().strip().split()[:2])
                nang[ni-1, 0] = idum
                for k in range(1,idum+1):
                     rs_ang[ni-1, 0, k-1], xi_ang[ni-1, 0, k-1], eta_ang[ni-1, 0, k-1], lambda_ang[ni-1, 0, k-1] = map(float, f.readline().strip().split()[:4])
         else:
             raise ValueError('Unrecognizable Angular Symmetry Function Type')
    # Compute ACSF functions
    natom=len(label)
    acsf=[]
    acsf.clear()
    radial = np.zeros((natom, telem, radmax))
    angular = np.zeros((natom, telem*(telem+1)//2, angmax))
    for i in range(natom):
        telem_i = chemclas(label[i],telem,tipo)
        atom_acsf=[]
        atom_acsf.clear()
        for j in range(natom):
            if j != i:  
                telem_j = chemclas(label[j],telem,tipo)
                Zj = dictionaries.atomicnumber[label[j]]
                factor = atomfactor(Zj)
                rij = coord[j] - coord[i]  
                distij = distance(coord[i], coord[j])
                fcutij = cutoff(rcut_rad, distij)
                if type_rad == 1:
                    for p in range(nrad[telem_i-1, telem_j-1]):
                        fpepe = np.exp(-eta_rad[telem_i-1, telem_j-1, p]*(distij-rs_rad[telem_i-1, telem_j-1, p])**2)*fcutij
                        radial[i, telem_j-1, p] += fpepe
                elif type_rad == 2:
                    for p in range(nrad[telem_i-1, 0]):
                        fpepe = factor*np.exp(-eta_rad[telem_i-1, 0, p]*(distij-rs_rad[telem_i-1, 0, p])**2)*fcutij
                        radial[i, 0, p] += fpepe
            for k in range(j+1,natom):
                if i != j and j != k and i != k:
                    rij = coord[j] - coord[i]
                    rik = coord[k] - coord[i]
                    rjk = coord[k] - coord[j]
                    dot = np.dot(rij, rik)
                    distij = np.linalg.norm(coord[i] - coord[j])
                    distik = np.linalg.norm(coord[i] - coord[k])
                    distjk = np.linalg.norm(coord[j] - coord[k])
                    fcutij = cutoff(rcut_ang,distij)
                    fcutik = cutoff(rcut_ang,distik)
                    fcutjk = cutoff(rcut_ang,distjk)
                    theta = np.arccos(dot / (distij * distik))
                    if ((type_ang == 1) or (type_ang == 2) or (type_ang == 3)):
                        telem_j = chemclas(label[j], telem, tipo)
                        telem_k = chemclas(label[k], telem, tipo)
                        neigh_id = 0
                        if (telem_k >= telem_j):
                            neigh = indjk[telem_j-1] + telem_k
                        else:
                            neigh = indjk[telem_k-1] + telem_j
                        for u in range(telem*(telem+1)//2):
                            if (vecino[u] == neigh):
                                neigh_id = u+1
                        if (neigh_id == 0):
                            raise Exception('FATAL ERROR IDENTIFYING THE NEIGHBOR PAIR NEIGH_ID=0')
                    elif (type_ang == 4):
                        Zj = dictionaries.atomicnumber[label[j]]
                        Zk = dictionaries.atomicnumber[label[k]]
                        factor = pairfactor(Zj, Zk)
                    if (type_ang == 1):
                        for p in range(nang[telem_i-1, neigh_id-1]):
                            fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                     (((1 + lambda_ang[telem_i-1, neigh_id-1, p]*np.cos(theta))**xi_ang[telem_i-1, neigh_id-1, p]) * 
                                      np.exp(-eta_ang[telem_i-1, neigh_id-1, p]*((distij-rs_ang[telem_i-1, neigh_id-1, p])**2 + 
                                                                            (distik-rs_ang[telem_i-1, neigh_id-1, p])**2 + 
                                                                            (distjk-rs_ang[telem_i-1, neigh_id-1, p])**2))) * 
                                     fcutij * fcutik * fcutjk)
                            angular[i, neigh_id-1, p] += fpepe
                    elif (type_ang == 2):
                        for p in range(nang[telem_i-1, neigh_id-1]):
                            fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                     (((1 + lambda_ang[telem_i-1, neigh_id-1, p]*np.cos(theta))**xi_ang[telem_i-1, neigh_id-1, p]) * 
                                      np.exp(-eta_ang[telem_i-1, neigh_id-1, p]*((distij-rs_ang[telem_i-1, neigh_id-1, p])**2 + 
                                                                            (distik-rs_ang[telem_i-1, neigh_id-1, p])**2))) * 
                                     fcutij * fcutik)
                            angular[i, neigh_id-1, p] += fpepe
                    elif (type_ang == 3):
                        for p in range(nang[telem_i-1, neigh_id-1]):
                            fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                     (((1 + np.cos(theta-theta_s[telem_i-1, neigh_id-1, p])) * xi_ang[telem_i-1, neigh_id-1, p]) * 
                                      np.exp(-eta_ang[telem_i-1, neigh_id-1, p]*((((distij+distik)/2)-rs_ang[telem_i-1, neigh_id-1, p])**2))) * 
                                     fcutij * fcutik)
                            angular[i, neigh_id-1, p] += fpepe
                    elif (type_ang == 4):
                        neigh_id = 1  
                        for p in range(nang[telem_i-1, neigh_id-1]):
                            fpepe = factor * (2.0**(1.0 - xi_ang[telem_i-1, neigh_id-1, p])) * \
                                (((1.0 + lambda_ang[telem_i-1, neigh_id-1, p] * np.cos(theta))**xi_ang[telem_i-1, neigh_id-1, p]) *
                                 np.exp(-eta_ang[telem_i-1, neigh_id-1, p] * ((distij - rs_ang[telem_i-1, neigh_id-1, p])**2 + 
                                 (distik - rs_ang[telem_i-1, neigh_id-1, p])**2 + (distjk - rs_ang[telem_i-1, neigh_id-1, p])**2))) * \
                                fcutij * fcutik * fcutjk
                            angular[i, 0, p] += fpepe
        if type_rad == 1:
           rad_acsf= [radial[i, k, p] for k in range(telem) for p in range(nrad[telem_i-1, k])]
        elif type_rad == 2:
           rad_acsf= [radial[i, 0, p] for p in range(nrad[telem_i-1, 0])]
        if type_ang in [1, 2, 3]:
           ang_acsf = [(angular[i, k, p]) for k in range(telem*(telem+1)//2) for p in range(nang[telem_i-1, k])]
        elif type_ang == 4:
           ang_acsf = [(angular[i, 0, p]) for p in range(nang[telem_i-1, 0])]
        # Create final list and list of lists
        for val in rad_acsf: atom_acsf.append(val)
        for val in ang_acsf: atom_acsf.append(val)
        acsf.append(atom_acsf)
    return acsf

def sfc_calc_for_database(label,coord,telem,tipo,type_rad,rcut_rad,radmax,nrad,rs_rad,eta_rad,indjk,vecino,type_ang,rcut_ang,angmax,rs_ang,xi_ang,eta_ang,lambda_ang,nang):
    """
    Compute ACSF functions to be stored in the database 
    """
    natom=len(label)
    acsf=[]
    acsf.clear()
    radial = np.zeros((natom, telem, radmax))
    angular = np.zeros((natom, telem*(telem+1)//2, angmax))
    for i in range(natom):
        telem_i = chemclas(label[i],telem,tipo)
        atom_acsf=[]
        atom_acsf.clear()
        for j in range(natom):
            if j != i:  
                telem_j = chemclas(label[j],telem,tipo)
                Zj = dictionaries.atomicnumber[label[j]]
                factor = atomfactor(Zj)
                rij = coord[j] - coord[i]  
                distij = distance(coord[i], coord[j])
                fcutij = cutoff(rcut_rad, distij)
                if type_rad == 1:
                    for p in range(nrad[telem_i-1, telem_j-1]):
                        fpepe = np.exp(-eta_rad[telem_i-1, telem_j-1, p]*(distij-rs_rad[telem_i-1, telem_j-1, p])**2)*fcutij
                        radial[i, telem_j-1, p] += fpepe
                elif type_rad == 2:
                    for p in range(nrad[telem_i-1, 0]):
                        fpepe = factor*np.exp(-eta_rad[telem_i-1, 0, p]*(distij-rs_rad[telem_i-1, 0, p])**2)*fcutij
                        radial[i, 0, p] += fpepe
            for k in range(j+1,natom):
                if i != j and j != k and i != k:
                    rij = coord[j] - coord[i]
                    rik = coord[k] - coord[i]
                    rjk = coord[k] - coord[j]
                    dot = np.dot(rij, rik)
                    distij = np.linalg.norm(coord[i] - coord[j])
                    distik = np.linalg.norm(coord[i] - coord[k])
                    distjk = np.linalg.norm(coord[j] - coord[k])
                    fcutij = cutoff(rcut_ang,distij)
                    fcutik = cutoff(rcut_ang,distik)
                    fcutjk = cutoff(rcut_ang,distjk)
                    theta = np.arccos(dot / (distij * distik))
                    if ((type_ang == 1) or (type_ang == 2) or (type_ang == 3)):
                        telem_j = chemclas(label[j], telem, tipo)
                        telem_k = chemclas(label[k], telem, tipo)
                        neigh_id = 0
                        if (telem_k >= telem_j):
                            neigh = indjk[telem_j-1] + telem_k
                        else:
                            neigh = indjk[telem_k-1] + telem_j
                        for u in range(telem*(telem+1)//2):
                            if (vecino[u] == neigh):
                                neigh_id = u+1
                        if (neigh_id == 0):
                            raise Exception('FATAL ERROR IDENTIFYING THE NEIGHBOR PAIR NEIGH_ID=0')
                    elif (type_ang == 4):
                        Zj = dictionaries.atomicnumber[label[j]]
                        Zk = dictionaries.atomicnumber[label[k]]
                        factor = pairfactor(Zj, Zk)
                    if (type_ang == 1):
                        for p in range(nang[telem_i-1, neigh_id-1]):
                            fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                     (((1 + lambda_ang[telem_i-1, neigh_id-1, p]*np.cos(theta))**xi_ang[telem_i-1, neigh_id-1, p]) * 
                                      np.exp(-eta_ang[telem_i-1, neigh_id-1, p]*((distij-rs_ang[telem_i-1, neigh_id-1, p])**2 + 
                                                                            (distik-rs_ang[telem_i-1, neigh_id-1, p])**2 + 
                                                                            (distjk-rs_ang[telem_i-1, neigh_id-1, p])**2))) * 
                                     fcutij * fcutik * fcutjk)
                            angular[i, neigh_id-1, p] += fpepe
                    elif (type_ang == 2):
                        for p in range(nang[telem_i-1, neigh_id-1]):
                            fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                     (((1 + lambda_ang[telem_i-1, neigh_id-1, p]*np.cos(theta))**xi_ang[telem_i-1, neigh_id-1, p]) * 
                                      np.exp(-eta_ang[telem_i-1, neigh_id-1, p]*((distij-rs_ang[telem_i-1, neigh_id-1, p])**2 + 
                                                                            (distik-rs_ang[telem_i-1, neigh_id-1, p])**2))) * 
                                     fcutij * fcutik)
                            angular[i, neigh_id-1, p] += fpepe
                    elif (type_ang == 3):
                        for p in range(nang[telem_i-1, neigh_id-1]):
                            fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                     (((1 + np.cos(theta-theta_s[telem_i-1, neigh_id-1, p])) * xi_ang[telem_i-1, neigh_id-1, p]) * 
                                      np.exp(-eta_ang[telem_i-1, neigh_id-1, p]*((((distij+distik)/2)-rs_ang[telem_i-1, neigh_id-1, p])**2))) * 
                                     fcutij * fcutik)
                            angular[i, neigh_id-1, p] += fpepe
                    elif (type_ang == 4):
                        neigh_id = 1  
                        for p in range(nang[telem_i-1, neigh_id-1]):
                            fpepe = factor * (2.0**(1.0 - xi_ang[telem_i-1, neigh_id-1, p])) * \
                                (((1.0 + lambda_ang[telem_i-1, neigh_id-1, p] * np.cos(theta))**xi_ang[telem_i-1, neigh_id-1, p]) *
                                 np.exp(-eta_ang[telem_i-1, neigh_id-1, p] * ((distij - rs_ang[telem_i-1, neigh_id-1, p])**2 + 
                                 (distik - rs_ang[telem_i-1, neigh_id-1, p])**2 + (distjk - rs_ang[telem_i-1, neigh_id-1, p])**2))) * \
                                fcutij * fcutik * fcutjk
                            angular[i, 0, p] += fpepe
        if type_rad == 1:
           rad_acsf= [radial[i, k, p] for k in range(telem) for p in range(nrad[telem_i-1, k])]
        elif type_rad == 2:
           rad_acsf= [radial[i, 0, p] for p in range(nrad[telem_i-1, 0])]
        if type_ang in [1, 2, 3]:
           ang_acsf = [(angular[i, k, p]) for k in range(telem*(telem+1)//2) for p in range(nang[telem_i-1, k])]
        elif type_ang == 4:
           ang_acsf = [(angular[i, 0, p]) for p in range(nang[telem_i-1, 0])]
        # Create final list and list of lists
        for val in rad_acsf: atom_acsf.append(val)
        for val in ang_acsf: atom_acsf.append(val)
        acsf.append(atom_acsf)
    return acsf
