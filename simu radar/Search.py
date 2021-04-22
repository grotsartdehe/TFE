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
import pandas as pd

f_s = 3.413e6;
f_0=24e9;
N_s=256;
#f_r=22.1 ;
c = 3e8;
w_0 = 2*np.pi*f_0;
BW = 250e6;
"""
X = np.array([[1,5,8,4],[4,8,3,9],[4,5,6,5]])
print(X.argsort())"""
def getzeroed(Z,row,col,classcar=0,res_d = 0.274):
    large = int(np.floor(5.4/res_d)/2) #longeur voiture ~ 4m + 1m pour marge
    
    Z[row-large:row+large,col-large:col+large]=0
    return Z


def Searchangle(Z,ambphi=0.5681818181818182,ambtheta=0.3125):
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
    
    # plt.figure()
    # plt.contourf(X[int(row//2):l0,int(col//2):l3],Y[int(row//2):l0,int(col//2):l3],m)
    x = np.max(m)
    # print(np.where(Z==x))
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
    return theta,phi

def plotDV(Z):
    dmax = 70
    vmax = 70/3.6
    res_d = 0.274
    res_v = 0.63/3.6
    
    
    N=256
    X = np.linspace(-vmax, vmax, N)
    Y = np.linspace(0, dmax, N)
    X, Y = np.meshgrid(X, Y)
    
    plt.xlim((-vmax,vmax))
    plt.contourf(X,Y,Z)
    plt.colorbar()
    plt.show()
    
def plotAngles(Z):
    X = np.linspace(-1,1,Z.shape[0])
    Y = np.linspace(-1,1,Z.shape[1])
    plt.contourf(X,Y,Z)
    plt.colorbar()
    plt.xlabel('cos(theta)')
    plt.ylabel('cos(phi)')
    plt.title('Heatmap (theta,phi)')
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
        if Z[x//row,x%row] >= 0.7:
            result.append(x)
        else: 
            cond = 1
        Z = getzeroed(Z,x//row,x%row)
        #plotDV(Z)
    if np.size(result)==0:
        return [], [],[],[]
    lignes,colonnes = np.unravel_index(result, (row,col))
    d = lignes* (c/(4*BW))
    v = (colonnes - col/2)*(c*np.pi*f_s/(2*w_0*N_s*256))
    return d,v,lignes,colonnes
    
# def Search(picklefile,folder):
#     infile = open(picklefile,'rb')
#     new_list = pickle.load(infile)
#     infile.close()

#     for i in new_list:
#         heatmapDV,heatmapsAngles = i.getdata()
#         count = i.getcounter()
#         """plotDV(heatmapDV)
#         """
#         #plotAngles(heatmapsAngles[2,:,:])
#         row = heatmapDV.shape[0]
#         col = heatmapDV.shape[1]
#         d,v = Searchdv(heatmapDV,row,col)
#         """lignes,colonnes = np.unravel_index(res, (row,col))
#         d = lignes* (c/(4*BW))
#         v = (colonnes - col/2)*(c*np.pi*f_s*3.6*2/(2*w_0*N_s*256))
#         name = 'data_'+ str(count).zfill(4)+ '.txt'
#         name = os.path.join(folder,name)
#         file = open(name,'w')
#         file.write('d v \n') 
#         for j in range(len(v)):
            
#             file.write(str(d[j]) +' ' +str(v[j]) + '\n')
#         file.close()"""
        
#     """Nouvelle methode """
# """def InsideLook(Z,dindex,vindex,res_d = 0.274):
#     large = int(np.floor(5/res_d)/2) #longeur voiture ~ 4m + 1m pour marge
#     dindex= int(dindex)
#     vindex = int(vindex)
#     m = Z[dindex-large:dindex+large,vindex-large:vindex+large]
#     maximum = np.max(m)
#     trouved,trouvev = np.where(Z==maximum)
#     if len(trouved) != 1:
#         mini = np.argmin((trouved - dindex)**2 + (trouvev - vindex)**2)
#         return trouved[mini], trouvev[mini]
#     else:
#         return trouved[0],trouvev[0]
    
    
# def LookDV(file,heatmapDV):
#     A = pd.read_csv(file,sep = ' ')
#     d = A['d']
    
#     v = A['v']
#     newdlist=[]
#     newvlist=[]
#     dindex =np.floor(d/(c/(4*BW)))
    
#     vindex =np.floor( v/(c*np.pi*f_s*3.6*2/(2*w_0*N_s*256)) + v.shape[0]/2)
#     for i in range(len(dindex)):
#         newdindex,newvindex = InsideLook(heatmapDV,dindex[i],vindex[i])
#         newd = newdindex* (c/(4*BW))
#         newv = (newvindex)*(c*np.pi*f_s*3.6*2/(2*w_0*N_s*256))
#         newdlist = np.append(newdlist,newd)
#         newvlist = np.append(newvlist,newv)
#     return newdlist,newvlist

# def association(d,v,cores):
#     assolist = []
#     for i in range(d.shape[0]):
#         dist = (d[i]-cores[:,0])**2 + (v[i]-cores[:,1])**2
#         indic = np.argmin(dist)
#         assolist = np.append(assolist,indic)
#         #d = numpy.delete(d, (indic), axis=0)
#         #v = numpy.delete(v, (indic), axis=0)
#     return assolist
        

# def Searchv2(picklefile,file):
#     infile = open(picklefile,'rb')
#     new_list = pickle.load(infile)
#     infile.close()

#     for i in new_list:
#         heatmapDV,heatmapsAngles = i.getdata()
#         count = i.getcounter()
#         cores = heatmapsAngles.getCores()
#         heatmapsAngles = heatmapsAngles.getZ()
#         newd,newv= LookDV(file,heatmapDV)
        
#         index = association(newd,newv,cores)
        
    
#         plotDV(heatmapDV)
        
#         #plotAngles(heatmapsAngles[2,:,:])

#      name = 'data_'+ str(count).zfill(4)+ '.txt'
#         name = os.path.join(folder,name)
#         file = open(name,'w')
#         file.write('d v \n') 
#         for j in range(len(v)):
            
#             file.write(str(d[j]) +' ' +str(v[j]) + '\n')
#         file.close()"""
    
    
    
# if __name__ == "__main__":
#     filename = 'bigfile1322'
#     file = 'test.txt'
#     folder = '/home/kdesousa/Documents/GitHub/TFE/Kalman/Data/radar/data-FX'
#     Searchv2(filename,file)
#     #Search(l,l.shape[0],l.shape[1])
    
    
    #index = Searchangle(heatmapsAngles,len(lignes))
        #costheta,cosphi = np.unravel_index(index,heatmapsAngles[0,:,:].shape[0],
                                        #heatmapsAngles[0,:,:].shape[1])
                                        
                                        
    