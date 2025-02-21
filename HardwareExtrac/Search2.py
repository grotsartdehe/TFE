#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:54:36 2021

@author: kdesousa
"""
import numpy as np
import pickle 
#from tools import *
import matplotlib.pyplot as plt
import math
import os 
import pandas as pd

f_s = 3.413e6;
f_0=24e9;
N_s=256;
#f_r=22.1 ;
c = 3e8;
w_0 = 2*np.pi*f_0;
BW = 545.5e6#250e6;
largeur = 3
res_d = 0.274
treshold = 50
"""
X = np.array([[1,5,8,4],[4,8,3,9],[4,5,6,5]])
print(X.argsort())"""
def getzeroed(Z,row,col):
    largeur = 5.4
    large = int(np.floor(largeur/res_d)/2)
    longeur = int(np.floor(largeur/res_d)/2)#longeur voiture ~ 4m + 1m pour marge
    a = row-large
    b = row+large
    c = col -longeur
    d = col +longeur
    if row - large<0 :
        a = 0
    if row + large >255:
        b = 256
    if col - longeur<0 :
        c = 0
    if col + longeur >255:
        d  = 256
    
    
        
    Z[a:b,c:d]=0
    return Z


def Searchangle(Z,ambphi,ambtheta,cam_number =0):
    result = []
    row = Z.shape[0]
    col = Z.shape[1]
    X = np.linspace(-1,1,row)
    Y  = np.linspace(-1,1,col)
    l0 = np.where(X<ambtheta)
    l3 = np.where(Y<ambphi)
    #X, Y = np.meshgrid(X,Y)
    
    l0 = l0[-1][-1]
    l3 = l3[-1][-1]
    
    Zreal = np.abs(Z)
    
    m = Zreal[int(row//2):l0,int(col//2):l3]

    mmax = np.max(m)
    
    
    # plt.figure()
    # #plt.contourf(X[int(row//2):l3],Y[int(col//2):l0],m/mmax)
    # plt.contourf(X,Y,20*np.log(Zreal))
    # plt.xlabel('u [ ]')
    # plt.ylabel('v [ ]')
    # plt.title('Carte thermique des angles pour cas réel')
    # plt.colorbar(label = 'Amplitude [dB]')

    #plt.contourf(X,Y,Z)
    x = np.max(m)
    
    #print(np.where(Z==x))
    ligne,colonne= np.where(Zreal==x)
    
    #ligne,colonne = np.unravel_index(x, (row,col))
    
    
    v = Y[ligne]
    u = X[colonne]
    
    
    #print(u*180/np.pi,v*180/np.pi)

   #theta = np.arccos(v)#np.arcsin(v)
   #phi = np.arccos(u)#np.arctan(u,np.sqrt(1-u**2-v**2))
    theta = np.arccos(v)
    phi = np.arctan2(u,np.sqrt(1-u**2-v**2))
    #print(theta*180/np.pi,phi*180/np.pi)
    # if cam_number == 1 or cam_number ==2:
    #     phi = np.arccos(u/np.sin(theta))
    # else:
    #     phi = np.arcsin(u/np.sin(theta))
    return theta,(phi)


    
    
def Searchdv(Z,row,col):
    result = []
    
    Z = np.array(Z)
    mean = np.mean(Z)
    var = np.var(Z)
    norm = np.max(Z)
    Z = Z/norm 
    
    cond = 0
    falseAlarm = 0
    while(cond == 0 and len(result)<10 and falseAlarm < treshold):
        x = np.argmax(Z)
        
        if Z[x//row,x%row] >= 0.50  :
            if checkside(Z,x//row,x%row):
                result.append(x)
            else:
                falseAlarm += 1
                
        else: 
            cond = 1
        
        Z = getzeroed(Z,x//row,x%row)
        #print(Z[x//row,x%row])
        #plotDV(Z)
    
    if np.size(result)==0 :
        return [], [],[],[]
    
    lignes,colonnes = np.unravel_index(result, (row,col))
    
    d = lignes* (c/(2*BW))
    v = (colonnes - col/2)*(c*np.pi*f_s/(2*w_0*N_s*256))
    
    return d,v,lignes,colonnes
    
def checkside(Z,row,col):
    "check les alentours de la heatmap"
    large = int(np.floor(largeur/res_d)/2)
    longeur = int(np.floor(largeur/res_d)/2)#longeur voiture ~ 4m + 1m pour marge
    a = row-large
    b = row+large
    c = col -longeur
    d = col +longeur
    
    if row - large<0 :
        a = 0
    if row + large >255:
        b = 256
    if col - longeur<0 :
        c = 0
    if col + longeur >255:
        d  = 256
    
    # print('mean',np.mean(Z[a:b,c:d]))
    # print('std',np.std(Z[a:b,c:d]))
    if np.mean(Z[a:b,c:d]) >0.2 and np.std(Z[a:b,c:d])>0.1:
        
        return True
    return False



def plotDV(Z):
    dmax = 70
    vmax = 70/3.6
    res_d = 0.274
    res_v = 0.63/3.6
    
    N=256
    d = np.arange(256)* (c/(4*BW))
    v = np.arange(-128,128,1)*(c*np.pi*f_s/(2*w_0*N_s*256))
    plt.figure()
    
    plt.contourf(v,d,Z)
    plt.colorbar()
    plt.xlabel('vitesse [m/s]')
    plt.ylabel('distance [m]')
    plt.title('Heatmap distance vitesse')
    plt.show()
    
def plotAngles(Z):
    plt.figure()
    X = np.linspace(-1,1,Z.shape[0])
    Y = np.linspace(-1,1,Z.shape[1])
    Zmax= np.max(Z)
    plt.contourf(X,Y,Z/Zmax)
    plt.colorbar()
    plt.xlabel('cos(phi)')
    plt.ylabel('cos(theta)')
    plt.title('Heatmap (theta,phi)')
    plt.show()

                                        
                                        
    