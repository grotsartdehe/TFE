#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:46:01 2021

@author: kdesousa
"""
"""
Created on Tue May 11 10:54:57 2021

@author: kdesousa
"""
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
from PIL import Image
from matplotlib import pyplot as plt
from threading import Thread
import pickle
import os, fnmatch

from Search2 import*
import os
import cv2
import pandas as pd
from correction import *


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

filename = '/home/kdesousa/Documents/GitHub/TFE/testcarrefour2/Prise1.pickle'
camfolder='/home/kdesousa/Documents/GitHub/TFE/testcarrefour2/Prise1'
filename = '/home/kdesousa/Documents/GitHub/TFE/testcarrefour/1620810275.395754.pickle'
camfolder='/home/kdesousa/Documents/GitHub/TFE/testcarrefour/1620810275.396608'
camfiles_dir = os.listdir(camfolder)
camfiles_dir.sort
camfiles = [ float(x.replace('.raw', '')) for x in camfiles_dir]
camfiles.sort()
infile = open(filename,'rb')
d = pickle.load(infile)
infile.close() 
d.sort(key=lambda a: a[1])

yolo = '/home/kdesousa/Documents/GitHub/TFE/HardwareExtrac/yolo/Prise1/labels'
def Zamb(lignes,colonnes,Z1,Z2,Z3):
    X = np.linspace(-1, 1, N)
    Y = np.linspace(-1, 1, N)
    ux,uy = np.meshgrid(X, Y)
    thetalist = []
    philist = []
    la = c/f_0
    ambiphi = la/(dx) #0.57246799 rad #32.8
    ambitheta = la/(dz)# 0.3176 rad #18.2
    
    adx = np.array([0 ,0,-1])*dx
    adz = np.array([-1,0,0])*dz
    
    large = int(np.floor(largeur/res_d)/2)
    for i in range(len(lignes)):
                   x1 = (Z1[lignes[i],colonnes[i]])
                   x2 = (Z2[lignes[i],colonnes[i]])
                   x3 =( Z3[lignes[i],colonnes[i]])
                   
                   k=np.pi * 2 * f_0/c #c(est la longuer d'onde)
                   
                   
                   op = x1*np.exp(-1j*k*(ux*adx[0]+uy*adz[0]))+\
                        x2*np.exp(-1j*k*(ux*adx[1]+uy*adz[1]))+\
                            x3*np.exp(-1j*k*(ux*adx[2]+uy*adz[2]))
                   
                   theta, phi = Searchangle(op,ambphi=ambiphi,ambtheta=ambitheta)
                   thetalist=np.append(thetalist,theta)
                   philist = np.append(philist,phi)
    return thetalist, philist

def init(picklecomplet):
    rip = []
    number = 3
    for i in picklecomplet:
        
        count = 0
        countcam = 0
        
        
        data_ok = i[0]
        z0 = array.array("H",data_ok)
        z = np.array(z0, dtype='complex')
        data = z[0::2] + 1j * z[1::2]
        data_cal = data[2:]
        
        rip = np.append(rip,data_cal)
        
        if rip.size >=(256*256*3*number):
                a1_cal = np.array([])
                a2_cal = np.array([])
                a3_cal = np.array([])
                
                for mm in range(number):
                    
                    a1_cal = np.append(a1_cal,rip[(0+mm*number)*256*256:256 * 256*(mm*number+1)])
                    
                    a2_cal = np.append(a2_cal,rip[(mm*number+1)*256 * 256:(mm*number+2) * 256 * 256])
                    a3_cal = np.append(a3_cal,rip[256 * 256 * (mm*number+2):(mm*number+3) * 256 * 256])
                    
                
                a1_cal = np.mean(a1_cal.reshape(number,256*256),axis=0).reshape(256,256)
                a2_cal = np.mean(a2_cal.reshape(number,256*256),axis=0).reshape(256,256)
                a3_cal = np.mean(a3_cal.reshape(number,256*256),axis=0).reshape(256,256)
                
                cal = np.abs(np.fft.fft2(a1_cal)**2 + np.fft.fft2(a2_cal )**2 +np.fft.fft2(a3_cal)**2 )
                
                fcal = np.fft.fftshift(cal, 0).T
                magn_cal = 20*np.log(np.abs(fcal))
                
                
                d = np.arange(256)* (c/(2*BW))
                v = np.arange(-128,128,1)*(c*np.pi*f_s/(2*w_0*N_s*256))
                
                
                # plt.figure()
                # plt.contourf(v,d,magn_cal)
                # plt.colorbar()
                # plt.title(str(count) + '        ' + str(i[1]))
                return a1_cal,a2_cal,a3_cal,magn_cal,rip
        count+=1 
def RealitySearch(pickle,a1_cal,a2_cal,a3_cal,magn_cal,rip):
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
    rip = np.append(rip,data_cal)
    
    if rip.size >=(256*256*3):
        mem = rip[0:3*256*256]
        rip = rip[3*256*256]
        Z1C = mem[0:256 * 256].reshape((256, 256))
        Z2C = mem[256 * 256:2 * 256 * 256].reshape((256, 256))
        Z3C = mem[256 * 256 * 2:3 * 256 * 256].reshape((256, 256))
        range_speed_ant1 = np.abs(np.fft.fft2(Z1C -a1_cal )**2+ np.fft.fft2(Z2C -a2_cal )**2+np.fft.fft2(Z3C -a3_cal ))
        range_speed_ant1 = np.abs(np.fft.fft2(Z1C )**2+ np.fft.fft2(Z2C  )**2+np.fft.fft2(Z3C )**2 )
        
        fshift = np.fft.fftshift(range_speed_ant1, 0).T
        magnitude_spectrum =20*np.log(np.abs(fshift))
        magn_norm1 = magnitude_spectrum - magn_cal
        
        magn_norm = (magn_norm1- np.mean(magn_norm1))/np.std(magn_norm1)
        d = np.arange(256)* (c/(2*BW))
        v = np.arange(-128,128,1)*(c*np.pi*f_s/(2*w_0*N_s*256))
        
        plt.figure()
        plt.contourf(v,d, magn_norm)
        
        #plt.title(str(count) +'   ' +  str(i[1]))
        plt.title('Carte thermique distance-vitesse pour cas réel')
        plt.xlabel('vitesse [m/s]')
        plt.ylabel('distance [m]')
        plt.colorbar(label = 'Amplitude [dB]')
        
        d,v,lignes,colonnes = Searchdv(magn_norm,256,256)
        
        theta,phi = Zamb(lignes, colonnes,Z1C.T- a1_cal.T,Z2C.T-a2_cal.T,Z3C.T-a3_cal.T)
        # im = np.array(Image.open(os.path.join(camfolder,str(name) + '.raw')))
        
        # plt.figure()
        # plt.imshow(im)
        # plt.title(str(name) + '.raw')
        
        if not(len(theta)==0 or len(phi)==0):
            lister = np.array([d,theta*180/pi,phi*180/pi,v]).T
            return lister,pickle[1],rip
        else :
            return [],pickle[1],rip
    else:
        return [],0,rip

def test(pickle): 
    data_ok = pickle[0]
    z0 = array.array("H",data_ok)
    z = np.array(z0, dtype='complex')
    
    data = z[0::2] + 1j * z[1::2]
    data_cal = data[2:]
    return data_cal

a1_cal,a2_cal,a3_cal,magn_cal,aaa = init(d)
count =0
countcam = 0
ci = 3
nn =300
rip = []
df = pd.read_csv('/home/kdesousa/Documents/GitHub/TFE/HardwareExtrac/data_final.csv',sep = ';',header=None).values
# for i in d:
#     if count==209 or count ==204 or  count == 211 or count == 214 or count == 218 or count == 219 or count == 221 or count ==224:
#     #if count < nn +30 and count >nn:
        
#         lister,time,rip= RealitySearch(i,a1_cal,a2_cal,a3_cal,magn_cal,rip)
    
#         while(time > camfiles[countcam]):
#             countcam +=1
        
#         if not len(lister) == 0:
            
#             # print(countcam)
#             # print()
#             # im = np.array(Image.open(os.path.join(camfolder,str(camfiles[countcam]).ljust(17,'0') + '.raw')))
#             # plt.figure()
#             # plt.imshow(im)
#             # plt.title(str(camfiles[countcam]) + '.raw    ' + str(count))
#             print(lister)
#             angles = pd.read_csv(os.path.join(yolo,'image_'+str(countcam).zfill(4)+'.txt'),header = None,sep = ' ').values
#             for jjj in range(len(angles)):
                
#                 if not angles[jjj,0]==9:
#                     v = np.zeros(8)
#                     print('angles')
#                     angle = angles[jjj,1:3]
#                     v[1] = 90  + (angle[1]*720 - 360)*28/1280
#                     v[2] = (angle[0]*1280-640)*28/1280
                
                
#                     for az in lister:
#                         v[5:7] = az[1:3]
                        
#                         print('old',v[5:7])
#                         print('cam',v[1:3])
                        
#                         v = correctionAngle(v,dx=dx,dy =dz)
#                         print('new',v[5:7])
#                         print()
#             # print(countcam)
#             # v = df[ci,:]
#             # ci +=1
#             # v = v[1::]
#             # for i in lister:
#             #     v[5:7] = i[1:3]
#             #     #print('old',v[5:7])
#             #     #print('cam',v[1:3])
                
#             #     v = correctionAngle(v,dx=dx,dy =dz)
#             #     print('new',v[5:7])
#     count+=1


count = 0
rip = []
number = 0
cal = 0
infile = open(filename,'rb')
d = pickle.load(infile)
infile.close() 
d.sort(key=lambda a: a[1])
for i in d:
    if count < 700:
        rip = np.append(rip,test(i))
        
        while( rip.size >(3*256*256)):
                
                mem = rip[0:3*256*256]
                rip = rip[3*256*256::]
                # if cal == 0:
                #     a1_cal = mem[0:256 * 256].reshape((256, 256))
                
                #     a2_cal = mem[256 * 256:2 * 256 * 256].reshape((256, 256))
                #     a3_cal = mem[256 * 256 * 2:3 * 256 * 256].reshape((256, 256))
                #     cal = np.abs(np.fft.fft2(a1_cal)**2 + np.fft.fft2(a2_cal )**2 +np.fft.fft2(a3_cal)**2 )
                #     fcal = np.fft.fftshift(cal, 0).T
                #     magn_cal = 20*np.log(np.abs(fcal))
                #     cal =1
                # else:
                Z1C = mem[0:256 * 256].reshape((256, 256))
                Z2C = mem[256 * 256:2 * 256 * 256].reshape((256, 256))
                Z3C = mem[256 * 256 * 2:3 * 256 * 256].reshape((256, 256))
                range_speed_ant1 = np.abs(np.fft.fft2(Z1C -a1_cal )**2+ np.fft.fft2(Z2C -a2_cal )**2+np.fft.fft2(Z3C -a3_cal ))
                #range_speed_ant1 = np.abs(np.fft.fft2(Z1C )**2+ np.fft.fft2(Z2C  )**2+np.fft.fft2(Z3C )**2 )
                
                fshift = np.fft.fftshift(range_speed_ant1, 0).T
                magnitude_spectrum =20*np.log(np.abs(fshift))
                magn_norm1 = magnitude_spectrum  - magn_cal
                
                magn_norm = (magn_norm1- np.mean(magn_norm1))/np.std(magn_norm1)
                d = np.arange(256)* (c/(2*BW))
                v = np.arange(-128,128,1)*(c*np.pi*f_s/(2*w_0*N_s*256))
                """ count ==145 exemple rapport"""
                if count >100 and count < 150 :
                    while(i[1] > camfiles[countcam]):
                        countcam +=1
                    
                    # im = np.array(Image.open(os.path.join(camfolder,str(camfiles[countcam]).ljust(17, '0') + '.raw')))
                    # plt.figure()
                    # plt.imshow(im)
                    # plt.title(str(camfiles[countcam]) + '.raw   ' + str(countcam))
                    
                    
                    d = np.arange(256)* (c/(2*BW))
                    v = np.arange(-128,128,1)*(c*np.pi*f_s/(2*w_0*N_s*256))
                    
                    plt.figure()
                    plt.contourf(v,d, magn_norm1)
                    
                    #plt.title(str(count) +'   ' +  str(i[1]))
                    plt.title('Carte thermique distance-vitesse ')
                    plt.xlabel('vitesse [m/s]')
                    plt.ylabel('distance [m]')
                    plt.colorbar(label = 'Amplitude [dB]')
                    
                    d,v,lignes,colonnes = Searchdv(magn_norm,256,256)
                    
                    theta,phi = Zamb(lignes , colonnes,Z1C.T- a1_cal.T,Z2C.T-a2_cal.T,Z3C.T-a3_cal.T)
                    
                    # if not(len(theta)==0 or len(phi)==0):
                    #     lister = np.array([d,theta*180/pi ,phi*180/pi,v]).T
                    #     v = np.zeros(8)
                    #     print(lister)
                    #     angles = pd.read_csv(os.path.join(yolo,'image_'+str(countcam).zfill(4)+'.txt'),header = None,sep = ' ').values
                    #     for jjj in range(len(angles)):
                    #         if not angles[jjj,0]==9:
                    #             print('angles')
                    #             angle = angles[jjj,1:3]
                    #             v[1] = 90  + (angle[1]*720 - 360)*28/1280
                    #             v[2] = (angle[0]*1280-640)*28/1280
                            
                            
                    #             for az in lister:
                    #                 v[5:7] = az[1:3]
                                    
                    #                 print('old',v[5:7])
                    #                 print('cam',v[1:3])
                                    
                    #                 v = correctionAngle(v,dx=dx,dy =dz)
                    #                 print('new',v[5:7])
                    #                 print()
                count +=1
            
count+=1




















"""backup"""


# init =0
# count = 0
# countcam = 0
# number = 0
# camname=[]
# condition = 0
# for i in d:
    
           
#     condition =0
    
#     if init == 0:
#         data_ok = i[0]
#         z0 = array.array("H",data_ok)
#         z = np.array(z0, dtype='complex')
#         data = z[0::2] + 1j * z[1::2]
#         data_cal = data[2:]
        
#         if data_cal.size >=(256*256*3):
#             a1_cal = data_cal[0:256 * 256].reshape((256, 256))
#             init = 1
#             a2_cal = data_cal[256 * 256:2 * 256 * 256].reshape((256, 256))
#             a3_cal = data_cal[256 * 256 * 2:3 * 256 * 256].reshape((256, 256))
#             cal = np.abs(np.fft.fft2(a1_cal)**2 + np.fft.fft2(a2_cal )**2 +np.fft.fft2(a3_cal)**2 )
#             fcal = np.fft.fftshift(cal, 0).T
#             magn_cal = 20*np.log(np.abs(fcal))
#     if  init ==1 and count > 383 and count <395:#and count==391:
        
#         data_ok = i[0]
#         z0 = array.array("H",data_ok)
#         z = np.array(z0, dtype='complex')
        
#         data = z[0::2] + 1j * z[1::2]
#         data_cal = data[2:]
#         if data_cal.size >=(256*256*3):
#             Z1C = data_cal[0:256 * 256].reshape((256, 256))
#             Z2C = data_cal[256 * 256:2 * 256 * 256].reshape((256, 256))
#             Z3C = data_cal[256 * 256 * 2:3 * 256 * 256].reshape((256, 256))
#             time = i[1] 
#             while( condition ==0):
                
#                 name = (camfiles[countcam])
#                 if time < name:
#                     condition =1
                    
#                 else: 
#                     countcam +=1
            
#             #range_speed_ant1 = np.abs(np.fft.fft2(Z1C - a1_cal)**2 + np.fft.fft2(Z2C - a2_cal)**2 +np.fft.fft2(Z3C - a3_cal)**2 )#- cal_ZC_1)
#             range_speed_ant1 = np.abs(np.fft.fft2(Z1C )**2+ np.fft.fft2(Z2C  )**2+np.fft.fft2(Z3C )**2 )
#             fshift = np.fft.fftshift(range_speed_ant1, 0).T
            
#             magnitude_spectrum =20*np.log(np.abs(fshift))
#             magn_norm1 = magnitude_spectrum - magn_cal
#             #magn_norm = magn_norm1/np.max(magn_norm1)
#             magn_norm = (magn_norm1 - np.mean(magn_norm1))/np.std(magn_norm1)
#             d = np.arange(256)* (c/(2*BW))
#             v = np.arange(-128,128,1)*(c*np.pi*f_s/(2*w_0*N_s*256))
#             plt.figure()
#             plt.contourf(v,d, magn_norm1)
#             plt.colorbar()
#             plt.title(str(count) +'   ' +  str(i[1]))
            
#             d,v,lignes,colonnes = Searchdv(magn_norm,256,256)
#             #print(lignes,colonnes)
#             theta,phi = Zamb(lignes, colonnes,Z1C.T- a1_cal.T,Z2C.T-a2_cal.T,Z3C.T-a3_cal.T)
#             im = np.array(Image.open(os.path.join(camfolder,str(name) + '.raw')))
            
#             plt.figure()
#             plt.imshow(im)
#             plt.title(str(name) + '.raw')
            
#             if not(len(theta)==0 or len(phi)==0):
#                 lister = np.array([d,theta*180/pi,phi*180/pi,v]).T
#                 print(lister)
#     count +=1

                            
                            
                            
# def synchronize(cam_folder,pickel):
#     list_cam = os.listdir(cam_folder)
    
#     list_cam= np.array([ float(x.replace('.raw', '')) for x in list_cam],dtype = np.float64 )
#     cam_arg = np.argsort(list_cam)
#     list_cam = list_cam[cam_arg]
#     list_rad = []
#     for i in pickel:
#         list_rad.append(i[1])
#         print( time.localtime( i[1] ))
#     list_rad = np.array(list_rad,dtype = np.float64)
#     rad_arg = np.argsort(list_rad)
#     list_rad = list_rad[rad_arg]
#     list_rad = list_rad[5::]
#     j=0
#     new_list_cam =[]
#     for i in list_rad:
#         while(i-list_cam[j]>1e-02):
#             j+=1
        
#         new_list_cam.append(list_cam[j])
    
#     new_list_cam= [ str(x) + '.raw' for x in new_list_cam]
    
#     return new_list_cam,rad_arg
#cam_folder = '/home/kdesousa/Documents/GitHub/TFE/HardwareExtrac/testparking2'
#a,b = synchronize(cam_folder,d)
        