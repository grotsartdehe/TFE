#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:54:36 2021

@author: kdesousa
"""
import numpy as np
import pickle 
from tools import *
import matplotlib.pyplot as plt
import math
import os 

"""
X = np.array([[1,5,8,4],[4,8,3,9],[4,5,6,5]])
print(X.argsort())"""
def getzeroed(Z,row,col,classcar=0,res_d = 0.274):
    large = int(np.floor(5/res_d)/2) #longeur voiture ~ 4m + 1m pour marge
    
    Z[row-large:row+large,col-large:col+large]=0
    return Z


def Searchangle(Z,detec):
    result = []
    num = Z.shape[0]
    for i in range(detec):
        if i <= num:
            x = np.argmax(Z[i,:,:])
            result.append(x)
        else:
            Zzero = np.random.normal((Z[0,:,:].shape))
            x = np.argmax(Zzero)
            result.append(x)
    return result

def plotDV(Z):
    dmax = 80
    vmax = 70
    res_d = 0.274
    res_v = 0.63
    sigma_d = res_d 
    sigma_v = res_v
    
    N1=math.floor(dmax/res_d) #291
    N2= math.floor(vmax/res_v) #111
    N=np.max([N1, N2])
    X = np.linspace(-N/2*res_v, N/2*res_v, N)
    Y = np.linspace(0, dmax, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    plt.xlim((-vmax,vmax))
    plt.contourf(X,Y,Z)
    plt.colorbar()
    plt.show()
    
def plotAngles(Z):
    X = np.linspace(-1,1,Z.shape[0])
    Y = np.linspace(-1,1,Z.shape[1])
    plt.contourf(X,Y,Z)
    plt.colorbar()
    plt.show()
    
def Searchdv(Z,row,col):
    result = []
    Z = np.array(Z)
    norm = np.max(Z)
    Z = Z/norm
    
    cond = 0
    
    while(cond == 0):
        x = np.argmax(Z)
        #print(Z[x//row,x%row])
        if Z[x//row,x%row] >= 0.5:
            result.append(x)
        else: 
            cond = 1
        Z = getzeroed(Z,x//row,x%row)
        plotDV(Z)
    return result
    
def Search(picklefile,folder):
    infile = open(picklefile,'rb')
    new_list = pickle.load(infile)
    infile.close()
    f_s = 3.413e6;
    f_0=24e9;
    N_s=256;
    f_r=22.1 ;
    c = 3e8;
    w_0 = 2*np.pi*f_0;
    BW = 250e6;
    for i in new_list:
        heatmapDV,heatmapsAngles = i.getdata()
        count = i.getcounter()
        """plotDV(heatmapDV)
        plotAngles(heatmapsAngles[0,:,:])"""
        row = heatmapDV.shape[0]
        col = heatmapDV.shape[1]
        res = Searchdv(heatmapDV,row,col)
        lignes,colonnes = np.unravel_index(res, (row,col))
        d = lignes* (c/(4*BW))
        v = (colonnes - col/2)*(c*np.pi*f_s*3.6*2/(2*w_0*N_s*256))
        name = 'data_'+ str(count).zfill(4)+ '.txt'
        name = os.path.join(folder,name)
        file = open(name,'w')
        file.write('d v \n') 
        for j in range(len(v)):
            
            file.write(str(d[j]) +' ' +str(v[j]) + '\n')
        file.close()
        
    
    
if __name__ == "__main__":
    filename = 'bigfile1322'
    folder = '/home/kdesousa/Documents/GitHub/TFE/Kalman/Data/radar/data-FX'
    Search(filename,folder)
    #Search(l,l.shape[0],l.shape[1])
    
    
    #index = Searchangle(heatmapsAngles,len(lignes))
        #costheta,cosphi = np.unravel_index(index,heatmapsAngles[0,:,:].shape[0],
                                        #heatmapsAngles[0,:,:].shape[1])
                                        
                                        
    