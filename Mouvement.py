# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:08:10 2020

@author: Gauthier_Rotsart
"""


import numpy as np
#Mouvement rectiligne uniforme: d = vt
def MRU(v,tmax):
    N = 10#nombre d'itérations
    x = 10
    y = np.zeros(N)
    d = np.zeros(N)
    t = np.linspace(0,tmax,N)
    for i in range(N):
        d[i] = np.sqrt((v*t[i])**2 + x**2)
    V = np.ones(N)*v
    return d,V
    