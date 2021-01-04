# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:19:52 2020

@author: Gauthier_Rotsart
"""
import numpy as np
from SimuRadar import radar
from Geometrie import voiture
d = 10 #distance du centre de masse
v = 30 #vitesse (en km/h)
theta = np.pi/6

dmax = 80
vmax = 70
res_d = 0.274
res_v = 0.63

#description du véhicules en prenant des valeurs +- réalistes
a = 1.7#largeur du vehicule
b = 4#longueur du vehicule

N = 1000#nombre de points
(x,y,m) = voiture(d,theta,a,b,N)
d_spec = np.sqrt(x**2 + y**2)
#♦radar(d_spec,v,theta,b,res_d,res_v,dmax,vmax)

