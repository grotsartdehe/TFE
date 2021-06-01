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
from Search2 import*
import os
import cv2
from validation import validation_cam, coordonnee_sph
from correction import correctionAngle

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
                   x1 = Z1[colonnes[i],lignes[i]]#(Z1[lignes[i],colonnes[i]])
                   x2 = Z2[colonnes[i],lignes[i]]#(Z2[lignes[i],colonnes[i]])
                   x3 = Z3[colonnes[i],lignes[i]]#(Z3[lignes[i],colonnes[i]])
                   
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
dz = 0.4#0.039351496918325776
dx = 0.022#0.021835281826631987

N=256
X = np.linspace(-1, 1, N)
Y = np.linspace(-1, 1, N)
ux,uy = np.meshgrid(X, Y)

filename = 'C:/Users/Gauthier_Rotsart/Documents/yolov5/data/images/testcarrefour/Test3.pickle'#'/home/kdesousa/Documents/GitHub/TFE/testcarrefour/1620810275.395754.pickle'
camfolder='C:/Users/Gauthier_Rotsart/Documents/yolov5/data/images/testcarrefour/Test3'
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
        #plot des heatmaps DV a chaque instant
        """
        plt.figure()
        plt.contourf(v,d, magn_norm1)
        plt.colorbar()
        #plt.title(str(count) +'   ' +  str(i[1]))
        plt.xlabel('vitesse [m/s]')
        plt.ylabel('distance [m]')
        plt.title('Heatmap distance vitesse')
        """
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

def vehicle(data_rad):
    d_rad = []
    theta_rad = []
    phi_rad = []
    v_rad = []
    for i in range(len(data_rad[:,3])):
        if np.abs(data_rad[i,3]) > 1 and np.abs(data_rad[i,0])>10:
            d_rad = np.append(d_rad,data_rad[i,0])
            theta_rad = np.append(theta_rad,data_rad[i,1])
            phi_rad = np.append(phi_rad,data_rad[i,2])
            v_rad = np.append(v_rad,data_rad[i,3])
    return d_rad,theta_rad,phi_rad,v_rad

def association(data_cam, d_rad, theta_rad, phi_rad, v_rad):
    if len(data_cam) < len(d_rad):
        data_cam_new = data_cam.copy()
        d_rad_new = np.zeros(len(data_cam))
        theta_rad_new = np.zeros(len(data_cam))
        phi_rad_new = np.zeros(len(data_cam))
        v_rad_new = np.zeros(len(data_cam))
        for i in range(len(data_cam)):
            L = np.sqrt((data_cam[i,0]-d_rad[:])**2).tolist()
            index = L.index(min(L))
            d_rad_new[i] = d_rad[index]
            theta_rad_new[i] = theta_rad[index]
            phi_rad_new[i] = phi_rad[index]
            v_rad_new[i] = v_rad[index]
    else:
        data_cam_new = np.zeros((len(d_rad),6))
        d_rad_new = d_rad.copy()
        theta_rad_new = theta_rad.copy()
        phi_rad_new = phi_rad.copy()
        v_rad_new = v_rad.copy()
        for i in range(len(d_rad)):
            L = np.sqrt((d_rad[i]-data_cam[:,0])**2).tolist()
            index = L.index(min(L))
            data_cam_new[i,:] = data_cam[index,:]
    return data_cam_new, d_rad_new, theta_rad_new, phi_rad_new, v_rad_new
            
a1_cal,a2_cal,a3_cal,magn_cal = init(d)
count =0#0
d_cam = []
theta_cam = []
phi_cam = []
v_cam = []
d_rad_new = []
theta_rad_new = []
phi_rad_new = []
v_rad_new = []
for i in d:
    #if count >= 2 and count <30:
    #if(count==418):
    lister,time_rad= RealitySearch(i,a1_cal,a2_cal,a3_cal,magn_cal)
    L = np.sqrt((time_rad - np.array(camfiles[:]))**2).tolist()
    index = L.index(min(L))
    time_cam = camfiles[index]
    #print(count,index)
    print('index',index, count)
    if index >=120 and index <= 170:
        if len(lister) != 0:
            #print(lister)
            d_rad,theta_rad,phi_rad,v_rad = vehicle(lister)
            data_val, data_check = validation_cam(index)
            data_val, d_rad, theta_rad, phi_rad, v_rad = association(data_val, d_rad,theta_rad,phi_rad,v_rad)
            if(len(d_rad) != 0 and len(data_val) != 0):
                for i in range(len(d_rad)):
                    diff1 = np.abs(data_val[i,0]-d_rad[i])
                    diff2 = np.abs(data_check[i,0]-d_rad[i])
                    if diff2 < diff1:
                        data_val = data_val
                    vect = np.array([data_val[i,0],data_val[i,1],data_val[i,2],d_rad[i],theta_rad[i],phi_rad[i],v_rad[i]])
                    data_corr = correctionAngle(vect)
                    #print('corr',data_corr)
                    d_cam = np.append(d_cam,data_corr[0])
                    theta_cam = np.append(theta_cam,data_corr[1])
                    phi_cam = np.append(phi_cam,data_corr[2])
                    v_cam = np.append(v_cam,data_val[i,3])
                    d_rad_new = np.append(d_rad_new,data_corr[3])
                    theta_rad_new = np.append(theta_rad_new,data_corr[4])
                    phi_rad_new = np.append(phi_rad_new,data_corr[5])
                    v_rad_new = np.append(v_rad_new,data_corr[6])
    elif count > 250:
        break
    count+=1
x_cam, y_cam, z_cam = coordonnee_sph(d_cam, theta_cam, phi_cam)
x_rad, y_rad, z_rad = coordonnee_sph(d_rad_new, theta_rad_new, phi_rad_new)
plt.figure()
plt.scatter(x_cam,y_cam)
plt.scatter(x_rad,y_rad)
plt.legend(["camera","radar"])










