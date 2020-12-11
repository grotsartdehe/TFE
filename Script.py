# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:19:52 2020

@author: Gauthier_Rotsart
"""
import numpy as np
from SimuRadar import radar
from Geometrie import voiture
from Mouvement import MRU
#d = MRU(#50 #distance du centre de masse
V = 30 #vitesse (en km/h)
d,v = MRU(V,10)
theta = np.pi/6

dmax = 150
vmax = 150
res_d = 0.274
res_v = 0.63

#description du véhicules en prenant des valeurs +- réalistes
a = 1.7#largeur du vehicule
b = 4#longueur du vehicule

N = 1000#nombre de points
for i in range(len(d)):
    (x,y) = voiture(d[i],theta,a,b,N)
    d_spec = np.sqrt(x**2 + y**2)
    #radar(d_spec,v[i],theta,b,res_d,res_v,dmax,vmax)


