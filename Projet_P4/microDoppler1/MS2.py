#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:36:15 2021

@author: kdesousa
"""
from scipy.io import loadmat
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from Search import Searchdv,plotDV


cal = loadmat('/home/kdesousa/Documents/GitHub/TFE/Projet_P4/microDoppler1/calibration.mat');
a_1_cal = cal['a_1'];

a_2_cal = cal['a_2'];
a_3_cal = cal['a_3'];
f_s = 3.413e6;
f_0=24e9;
N_s=256;
f_r=22.1 ;
c = 3e8;
w_0 = 2*pi*24e9;
BW = 250e6;

directory = os.listdir('/home/kdesousa/Documents/GitHub/TFE/Projet_P4/microDoppler1')
directory.sort()
z=0
dx = 36e-3
dy = 22.5e-3
N=256
X = np.linspace(-1, 1, N)
Y = np.linspace(-1, 1, N)
ux,uy = np.meshgrid(X, Y)

r_x= [0.036/2, -0.036/2, -0.036/2];
r_y= [-0.0225/2, -0.0225/2, 0.0225/2];
test = np.complex64(np.zeros(3*len(directory)))
j =0
for i in directory:
    
    if i.endswith('mat') and  not i.startswith('cali') and j == 60 :
        
        x = loadmat(i)
        x_a1 = x['a_1'] -a_1_cal
        x_a2 = x['a_2'] -a_2_cal
        x_a3 = x['a_3'] -a_3_cal
        maxi = np.argmax(x_a1)
        x1max = x_a1.reshape(x_a1.size)[maxi]
        x2max = x_a2.reshape(x_a1.size)[maxi]
        x3max = x_a3.reshape(x_a1.size)[maxi]
        test[z*3]= x1max
        test[z*3 +1]= x2max
        test[z*3 +2]= x3max
        lam=np.pi * 2 * f_0/c
        
        op= x1max*np.exp(-1j*lam*dy*uy)+\
            x2max+\
            x3max*np.exp(-1j*lam*dx*ux)
       
        plt.figure(clear=True)
        plt.contourf(X,Y,np.abs(op))
        plt.title('Heatmap of angles')
        plt.xlabel('ux [rad]')
        plt.ylabel('uy [rad]')
        plt.colorbar()
        
        
        name = 'figurescos/fig_'+ str(z)
        plt.savefig(name)
        plt.close()
        z +=1
        
    j+=1
        

