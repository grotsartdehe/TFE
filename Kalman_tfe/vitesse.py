#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:20:00 2021

@author: kdesousa
"""
import numpy as np
import math

#from kalman_graphics import kalman_draw
import os
import time
import datetime
import pickle


import pandas as pd
import matplotlib.pyplot as plt
def extract(df,pos_cam):
    """
    Parameters
    ----------
    dataframe : panda dataframe
    Returns
    -------
    d : liste of distance.
    v : liste de vitesse
    theta : angle azimutale
    phi : angle d'élevationmain.py
    
    """
    Xpos1 = (df['XPos']-pos_cam[1])/100
    Ypos1 = (df['YPos']-pos_cam[2])/100
    Zpos1 = (df['ZPos']-pos_cam[3])/100
    Xpos2D = df['2D_XPos'].values
    Ypos2D = df['2D_YPos'].values
    pitch =  -pos_cam[5]*np.pi/180
    
    yaw = -pos_cam[6]*np.pi/180
    Rz =np.array( [[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    Ry = np.array([[np.cos(pitch),0, np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    R = Rz@Ry
    
    #Posnew =  R @  np.array([Xpos1,Ypos1,Zpos1])
    Posnew =  np.array([Xpos1,Ypos1,Zpos1])
    Xpos = Posnew[0,:]
    Ypos = Posnew[1,:]
    Zpos = Posnew[2,:]
    d = np.sqrt(Xpos**2 + Ypos**2 + Zpos**2)
    
    #normaliser pour obtenir vecteur cam-vehicule
    Xposdir = Xpos/d
    Yposdir = Ypos/d
    Zposdir = Zpos/d
    W = 1920
    H = 1280
    cond2 = (Xpos2D >= 0) & (Xpos2D <= W) & (Ypos2D >= 0) & (Ypos2D <= H)
    Xpos2D = Xpos2D[cond2]
    Ypos2D = Ypos2D[cond2]
    cond = (d < 70) * cond2
    
    
    v = df['Vel']/100
    
    Xdir = df['XDir']
    Ydir = df['YDir']
    Zdir= df['ZDir']
    
    #Dirnew =  R @  np.array([Xdir,Ydir,Zdir])
    Dirnew =  np.array([Xdir,Ydir,Zdir])
    Xdir = Dirnew[0]
    Ydir = Dirnew[1]
    Zdir= Dirnew[2]
    
    
    #projection orthogonale
    norm =  np.sqrt(Xposdir**2 + Yposdir**2 + Zposdir**2) 
    Vdir = (Xdir*Xposdir + Ydir*Yposdir+Zdir*Zposdir)/norm # diviser par norm = 1
    
    
    #print(np.arccos(Vdir/v)*180/np.pi)
   
    
    theta = np.arccos(Zpos/d) 
    phi = np.arctan2(Ypos,Xpos)
    
    classcar = df['ID']
    v1 = v*Vdir
    
    xsi = np.arctan2(Ydir,Xdir)
    #print(xsi*180/np.pi)
    #store = np.array([d[cond],theta[cond]*180/np.pi,phi[cond]*180/np.pi,v1[cond]]).T
    #print(store)
    
   
    
    return d[cond],v1[cond],theta[cond],phi[cond],classcar[cond],xsi[cond],v[cond]#+pi/2
    

csv_folder= '/home/kdesousa/Documents/GitHub/TFE/Kalman/2021_04_06_15_40_39_604/cam_00'
cam_number = int( csv_folder[-1])
pos_cam = os.path.join(csv_folder,'pos_cam_'+csv_folder[-2:]+'.csv')
df = pd.read_csv(pos_cam, sep =';')

pos_cam = df.values[1,:]#[df.iloc[2]['Xpos'],df.iloc[2]['Ypos'],df.iloc[2]['Zpos']]
Thelist = []
csv_data = os.listdir(csv_folder)
csv_data.sort()
counter = 581
for i in csv_data:
    if  not i.startswith('.~lock') and not i.startswith('pos') and not i.endswith('.jpg'):
    
        n = 1540
        if  counter == n :# or( counter > n and counter < n+20) :
             
            file = os.path.join(csv_folder,i)
            print(file)
            df = pd.read_csv(file,sep =';',index_col=False )
            v_abs = df['Vel']/100
            
            d_real,v_real,theta,phi,classcar,xsi,vabs = extract(df,pos_cam)
            v_abs = df['Vel']/100 
            v_abs = v_abs[v_real.index].values
            #print('vabs',vabs)
            #print('v_real',v_real)
            costheta= np.cos(theta)*np.cos(phi)
            
            x = np.arccos(costheta)
            print('vabs',vabs)
            print('v_test',v_real/(costheta))
            
        counter +=1 