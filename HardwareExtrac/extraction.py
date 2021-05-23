#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:54:57 2021

@author: kdesousa
"""
import pickle

import pickle 
from multiprocessing import Process, Queue
import time 
import numpy as np
import cv2
import datetime
import threading 
import socket
import struct
import array
from radc_frame import radar_frame
from matplotlib import pyplot as plt
from threading import Thread
import pickle
import os, fnmatch
from lib_radar import *
from Search import*
import os
import cv2
def Zamb(lignes,colonnes,Z1,Z2,Z3):
    X = np.linspace(-1, 1, N)
    Y = np.linspace(-1, 1, N)
    ux,uy = np.meshgrid(X, Y)
    thetalist = []
    philist = []
    la = c/f_0
    ambiphi = la/(dx) #0.57246799 rad #32.8
    ambitheta = la/(dz)# 0.3176 rad #18.2
    large = int(np.floor(largeur/res_d)/2)
    for i in range(len(lignes)):
                   x1 = (Z1[lignes[i],colonnes[i]])
                   x2 = (Z2[lignes[i],colonnes[i]])
                   x3 =( Z3[lignes[i],colonnes[i]])
                   
                   k=np.pi * 2 * f_0/c #c(est la longuer d'onde)
        
                   op= x3*np.exp(1j*k*dx*ux)+\
                        x2+\
                            x1*np.exp(-1j*k*dz*uy)
                   #plotAngles(op)
                   theta, phi = Searchangle(op,ambphi=ambiphi,ambtheta=ambitheta)
                   thetalist=np.append(thetalist,theta)
                   philist = np.append(philist,phi)
    return thetalist, philist

f_s = 3.413e6;
f0=24e9;
N_s=256;
f_r=22.1 ;
c = 3e8;
pi =np.pi
w_0 = 2*pi*f0;
BW = 545.5e6;
dz = 0.039351496918325776
dx = 0.021835281826631987

N=256
X = np.linspace(-1, 1, N)
Y = np.linspace(-1, 1, N)
ux,uy = np.meshgrid(X, Y)

filename = '/home/kdesousa/Documents/GitHub/TFE/testcarrefour/1620810275.395754.pickle'
camfolder='/home/kdesousa/Documents/GitHub/TFE/testcarrefour/1620810275.396608'
camfiles_dir = os.listdir(camfolder)
camfiles_dir.sort
camfiles = [ float(x.replace('.raw', '')) for x in camfiles_dir]
camfiles.sort()
infile = open(filename,'rb')
d = pickle.load(infile)
infile.close() 
def init(picklecomplet):
    
    for i in picklecomplet:
        
        count = 0
        countcam = 0
        number = 0
        
        data_ok = i[0]
        z0 = array.array("H",data_ok)
        z = np.array(z0, dtype='complex')
        data = z[0::2] + 1j * z[1::2]
        data_cal = data[2:]
        
        if data_cal.size >=(256*256*3):
            a1_cal = data_cal[0:256 * 256].reshape((256, 256))
            
            a2_cal = data_cal[256 * 256:2 * 256 * 256].reshape((256, 256))
            a3_cal = data_cal[256 * 256 * 2:3 * 256 * 256].reshape((256, 256))
            cal = np.abs(np.fft.fft2(a1_cal)**2 + np.fft.fft2(a2_cal )**2 +np.fft.fft2(a3_cal)**2 )
            fcal = np.fft.fftshift(cal, 0).T
            magn_cal = 20*np.log(np.abs(fcal))
            return a1_cal,a2_cal,a3_cal,magn_cal
def RealitySearch(pickle,a1_cal,a2_cal,a3_cal,magn_cal):
    """
    

    Parameters
    ----------
    pickle : tuple (radardata, time)
        radardata contient un byte array avec les données radar captés au temps 
        time 
    aX_cal: calibration pour l'antenne X'
    magn_cal:calibration totale
    Returns
    -------
    None.

    """
    
    
        
    data_ok = pickle[0]
    z0 = array.array("H",data_ok)
    z = np.array(z0, dtype='complex')
    
    data = z[0::2] + 1j * z[1::2]
    data_cal = data[2:]
    if data_cal.size >=(256*256*3):
        Z1C = data_cal[0:256 * 256].reshape((256, 256))
        Z2C = data_cal[256 * 256:2 * 256 * 256].reshape((256, 256))
        Z3C = data_cal[256 * 256 * 2:3 * 256 * 256].reshape((256, 256))
        range_speed_ant1 = np.abs(np.fft.fft2(Z1C )**2+ np.fft.fft2(Z2C  )**2+np.fft.fft2(Z3C )**2 )
        fshift = np.fft.fftshift(range_speed_ant1, 0).T
        magnitude_spectrum =20*np.log(np.abs(fshift))
        magn_norm1 = magnitude_spectrum - magn_cal
        
        magn_norm = (magn_norm1 - np.mean(magn_norm1))/np.std(magn_norm1)
        d = np.arange(256)* (c/(2*BW))
        v = np.arange(-128,128,1)*(c*np.pi*f_s/(2*w_0*N_s*256))
        plt.figure()
        plt.contourf(v,d, magn_norm1)
        plt.colorbar()
        plt.title(str(count) +'   ' +  str(i[1]))
        
        d,v,lignes,colonnes = Searchdv(magn_norm,256,256)
        #print(lignes,colonnes)
        theta,phi = Zamb(lignes, colonnes,Z1C.T- a1_cal.T,Z2C.T-a2_cal.T,Z3C.T-a3_cal.T)
        # im = np.array(Image.open(os.path.join(camfolder,str(name) + '.raw')))
        
        # plt.figure()
        # plt.imshow(im)
        # plt.title(str(name) + '.raw')
        
        if not(len(theta)==0 or len(phi)==0):
            lister = np.array([d,theta*180/pi,phi*180/pi,v]).T
            return lister,pickle[1]
        else :
            return [],pickle[1]
    else:
        return [],0


a1_cal,a2_cal,a3_cal,magn_cal = init(d)
count =0
for i in d:
    if count >400 and count <450:
        lister,time= RealitySearch(i,a1_cal,a2_cal,a3_cal,magn_cal)
        print(lister)
    count+=1










