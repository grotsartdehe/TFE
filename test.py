# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:09:10 2021

@author: Gauthier_Rotsart
"""
import pandas as pd
import numpy as np
import os.path 
def hausdorf(num1,num2):
    W = 1920
    H = 1080
    data1 = pd.read_csv ('00_'+str(num1).zfill(8)+'.txt',sep=' ',header = None).values
    data2 = pd.read_csv ('00_'+str(num2).zfill(8)+'.txt',sep=' ',header = None).values
    #denormalisation des datas de yolo
    data1[:,1] = data1[:,1]*W#coord x du centre de masse
    data1[:,2] = data1[:,2]*H#coord y du centre de masse
    data1[:,3] = data1[:,3]*W#largeur du rectangle
    data1[:,4] = data1[:,4]*H#hauteur du rectangle
    
    data2[:,1] = data2[:,1]*W#coord x du centre de masse
    data2[:,2] = data2[:,2]*H#coord y du centre de masse
    data2[:,3] = data2[:,3]*W#largeur du rectangle
    data2[:,4] = data2[:,4]*H#hauteur du rectangle
    #print('ola1',data1[:,4])
    #print('ola',data2[:,4])
    cm1 = np.int32(np.array([data1[:,1].tolist(),data1[:,2].tolist()]).T)
    cm2 = np.int32(np.array([data2[:,1].tolist(),data2[:,2].tolist()]).T)
    #distance parcourue par chaque centre de masse
    distance = np.zeros((len(cm2),2))
    for i in range(len(cm2)):
        value = cm2[i]
        L = np.sqrt((value[0]-cm1[:,0])**2+(value[1]-cm1[:,1])**2).tolist()
        distance[i,0] = min(L)
        distance[i,1] = L.index(min(L))
    #changement de coordonnees
    cm1[:,0] = cm1[:,0]-960*np.ones(cm1.shape[0])
    cm1[:,1] = cm1[:,1]-540*np.ones(cm1.shape[0])
    cm2[:,0] = cm2[:,0]-960*np.ones(cm2.shape[0])
    cm2[:,1] = cm2[:,1]-540*np.ones(cm2.shape[0])
    #print('cm1',cm1)
    #print('cm2',cm2)
    #calcul de la distance camera/objet en pixel
    d = np.zeros((len(cm2)))
    for i in range((len(cm2))):
        index_i = np.int32(distance[i,1])
        delta_i = distance[i,0]#distance parcourue entre les deux frames
        print("delta:",delta_i)
        b = data2[i,4]#[i,1]#cm2[i,0]
        a = data1[index_i,4]#cm1[index_i,1]#cm1[index_i,0]
        print(a,b)
        if (b==a):
            b+=1
            d[i] = b*delta_i/(b-a)#np.abs(b*delta_i/(b-a))
            print('cc')
        else:
            d[i] = b*delta_i/(b-a)#np.abs(b*delta_i/(b-a))
    return data2[:,4],cm2

count = 2437#test de la frame 2437
points_X = []
points_Y = []
points_Z = []
Xc = 33442.0
Yc = -76290.0
Zc = -53546.0
while(os.path.isfile('00_'+str(count).zfill(8)+'.csv') == 1):
    #data = pd.read_csv('01_'+str(count).zfill(8)+'.csv', sep=';').values
    #distance_real = np.sqrt(np.float32((data[:,3]-Xc)**2+(data[:,4]-Yc)**2+(data[:,5]-Zc)**2))
    #print('distance_reelle',distance_real/100)
    H,cm2 = hausdorf(count-2,count)
    #cm2[:,0] = cm2[:,0]+960*np.ones(cm2.shape[0])
    #cm2[:,1] = cm2[:,1]+540*np.ones(cm2.shape[0])
    data = pd.read_csv('00_'+str(count).zfill(8)+'.csv', sep=';').values
    #print(data)
    point3D_test = data[:,3:6]-[Xc,Yc,Zc]
    print('pixel_reel_YOLO',cm2)
    beta = 28
    distance = np.zeros(len(H))
    print(H)
    for i in range(len(H)):
        theta = H[i]*beta/1080
        distance[i] = (1.72/(H[i]/2+cm2[i,1]))*925#1.72/np.tan(theta*np.pi/180)
    count+=1
distance_real = np.sqrt(np.float32((data[:,3]-Xc)**2+(data[:,4]-Yc)**2+(data[:,5]-Zc)**2))
print('distance_estimation',distance)
print('distance_reelle',distance_real/100)