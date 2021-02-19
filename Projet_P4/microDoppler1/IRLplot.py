#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:38:23 2021

@author: kdesousa
"""
from scipy.io import loadmat
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from matplotlib import animation


cal = loadmat('calibration.mat');
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

"""
x = loadmat('data_QCSR2018-10-31 10_55_47.190466.npz.mat')
test = np.array(x['a_1']-a_1_cal )
print(test.shape)
"""
directory = os.listdir('/home/kdesousa/Documents/GitHub/TFE/RadarIRL/microDoppler1')
directory.sort()
z=0
norm = 20
for i in directory:
    if i.endswith('mat') and  not i.startswith('cali') :
        x = loadmat(i)
        x_a1 = x['a_1'] -a_1_cal
        x_a2 = x['a_2'] -a_2_cal
        x_a3 = x['a_3'] -a_3_cal
        l = np.abs(np.fft.fft2(x_a1.T))+np.abs(np.fft.fft2(x_a2.T))+\
            np.abs(np.fft.fft2(x_a3.T))
        #if z < 10 and z>5:
        #plt.figure(clear=True)
        d = np.arange(0,x_a1.shape[1])*(c/(2*BW)) ; # -> range
        v = np.arange(-x_a1.shape[0]/2,x_a1.shape[0]/2)*(c*pi*f_s*3.6*2/(2*w_0*N_s*256));# -> doppler,
        plt.figure(clear=True)
        plt.contourf(v,d,np.log(np.abs(l))/norm)
        plt.colorbar() 
        plt.title('heatmap (d,v)')
        plt.xlabel('vitesse [km/h]')
        plt.ylabel('distance [m]')
        
        name = 'figures/fig_'+ str(z)
        plt.savefig(name)
        plt.close()
        z +=1
"""
m = np.fft.fft2(test)
plt.contourf(m)
plt.colorbar()
"""

    
"""
ig = plt.figure()

anim = FuncAnimation(fig, animation,frames=20)
#anim = FuncAnimation(fig, animate, init_func=init,
                           #    frames=200, interval=200, blit=True)
anim.save('demo.gif', writer='imagemagick')
"""





