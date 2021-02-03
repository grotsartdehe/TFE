# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:30:57 2020

@author: Gauthier_Rotsart
"""

import numpy as np
import matplotlib.pyplot as plt
import math

"""
Radar retourne la d et la vitesse d'un vehicule de largeur L 
"""
def radar(d,v,theta,L):
    #variation en distance
    dmax = 80
    vmax = 70
    res_d = 0.274
    res_v = 0.63
    sigma_d = res_d 
    sigma_v = res_v
    d_new = np.random.normal(d, sigma_d)
    v_new = np.random.normal(v,sigma_v)
    plt.figure()
    plt.scatter(np.array([v_new]),np.array([d_new]), marker = '.', c = np.array([50]))
    plt.xlim(-vmax, vmax)
    plt.ylim(0,dmax)
    plt.xlabel('vitesse [km/h]')
    plt.ylabel('distance [m]')
    
    plt.scatter(np.array([v]),np.array([d+L/2]), marker = '.', c = np.array([50]))
    plt.scatter(np.array([v]),np.array([d-L/2]), marker = '.', c = np.array([50]))
    
    omega = v*math.cos(theta)#vitesse radiale 
    #calcul d'amplitude, A MODIFIER
    a1 = [i+1 for i in range(int(omega/sigma_v)//2+1)]
    a2 = [int(omega/sigma_v) - 1 - i for i in range(int(omega/sigma_v)//2)]
    amplitude = np.append(np.array(a1), np.array(a2))#amplitude decroissante
    amp = np.ones(int(omega/sigma_v))*(50-15)
    
    vitesse = np.linspace(int(v-omega/2),int(v+omega/2),int(omega/sigma_v))#range de vitesse
    dist = np.ones(len(vitesse))*d
    plt.scatter(vitesse, dist,marker = '|', c = amp)
    cbar = plt.colorbar()
    cbar.set_label("Amplitude")


