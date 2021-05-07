#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:40:27 2021

@author: kdesousa
"""
import numpy as np
def amb(a,b,dx = 0.022,dy = 0.04,f0=24e9):
    theta_rad,phi_rad = a
    theta_cam,phi_cam = b
    
    
    
    lam = 3e8/f0 
    ambphi = lam*180/(dx*np.pi) #32.554420177887685 
    ambtheta = lam*180/(dy*np.pi)#17.904931097838226
    nx = np.rint((phi_cam - phi_rad)/ambphi)
    nz = np.rint((theta_cam - theta_rad)/ambtheta)
   
    phi_rad += (nx)*ambphi
    theta_rad += (nz)*ambtheta
    
    
    return theta_rad,phi_rad
camera = np.zeros((2,3))
radar = camera = np.zeros((2,3))
length = camera.shape[0] #np.min(camera.shape[0],radar.shape[0])
dist = np.zeros(length)
index = np.zeros(length)
for i in camera:
    for j in radar:
        angles = amb(camera[i,1:3],radar[j,1:3])
        distance = (angles[0] - camera[1])**2 + (angles[1] - camera[2])
        
        if dist[i]>distance:
            index[i]= j
            dist[i]=distance[i]

        
        

