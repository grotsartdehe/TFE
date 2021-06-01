# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:25:07 2021

@author: Gauthier_Rotsart
"""

import pandas as pd
import numpy as np
import os.path 
import csv
from main import CreateandSearch
from correction import correction
#from calibration import isVehicle, isTheSame

def isVehicle(data):
    compteur = 0
    for i in range(len(data)):
        if data[i,0] != 0 and data[i,0] != 9 and data[i,0] != 11 and data[i,0] != 12:
            compteur+=1
    data_new = np.empty((compteur,data.shape[1]))
    compteur = 0
    for i in range(len(data)):
        if data[i,0] != 0 and data[i,0] != 9 and data[i,0] != 11 and data[i,0] != 12:
            data_new[compteur,:] = data[i,:]
            compteur +=1
    return data_new

def isTheSame(data):
    data = data.tolist()
    data_new = data.copy()
    index = []
    for i in range(len(data)):
        #print(i)
        for j in range(len(data)):
            if i != j and index.count(i) == 0:
                #print(len(data_new))
                distance_i = np.sqrt((data[i][1]-data[j][1])**2+(data[i][2]-data[j][2])**2)
                if distance_i < 20:
                    data_new.remove(data[i][:])
                    index = np.append(index,np.append(i,j)).tolist()
                    #print(index)
    return np.array(data_new)

def coordonnee_sph(r,theta,phi):
    theta = np.radians(theta)
    phi = np.radians(phi)
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z

def dimension(cm1,cm2):
    dimension = np.zeros(len(cm2))
    for i in range(len(cm2)):
        value = cm2[i]
        L1 = np.sqrt(((cm2[i,0]-cm1[:,0].tolist())**2+(cm2[i,1]-cm1[:,1].tolist())**2)).tolist()
        index_i1 = L1.index(min(L1))
        delta_x = np.abs(cm1[index_i1,0]-cm2[i,0])
        delta_y = np.abs(cm1[index_i1,1]-cm2[i,1])
        if np.abs(delta_x) > np.abs(delta_y):#avance principalement verticalement dans l'image
            dimension[i] = 4.5
        elif np.abs(delta_x) < np.abs(delta_y):#avance principalement horizontalement dans l'image
            dimension[i] = 1.8
        else:#n'avance pas
            dimension[i] = 1.8
    return dimension

def calibration(L,cm,dimension,W,H):
    beta = 28
    distance = np.zeros(len(cm))
    theta = np.zeros(len(cm))
    phi = np.zeros(len(cm))
    x = np.zeros(len(cm))
    y = np.zeros(len(cm))
    z = np.zeros(len(cm))
    donnée = np.zeros((len(cm),6))
    pitch = 0
    yaw = 0
    value_copy = cm.copy()
    for i in range(len(cm)):
        #L1 = np.sqrt(((value_copy[i,0]-cm[:,0].tolist())**2+(value_copy[i,1]-cm[:,1].tolist())**2)).tolist()
        #index_i1 = L1.index(min(L1))
        #print('L',L1)
        theta1 = (cm[i,0]-W/2 - L[i]/2)*beta/W
        theta2 = (cm[i,0]-W/2)*beta/W
        
        distance[i] = (dimension[i]/(2))/(np.cos(theta2*np.pi/180)*(np.tan(theta2*np.pi/180)-np.tan(theta1*np.pi/180)))
        #dd = distanceModif(cm[i],dimension[i],round(L[i]),W,1080)
        distance[i] = np.sqrt(1.2**2+distance[i]**2)
        theta[i] = 90 - pitch + (cm[i,1]-H/2)*beta/W
        phi[i] = yaw + (cm[i,0]-W/2)*beta/W
        
        donnée[i,0] = distance[i]
        donnée[i,1] = theta[i]
        donnée[i,2] = phi[i]
        x[i],y[i],z[i] = coordonnee_sph(distance[i],theta[i],phi[i])
        donnée[i,3] = x[i]
        donnée[i,4] = y[i]
        donnée[i,5] = z[i]
        #print(index_i1)
        #print(value_copy)
        #value_copy = value_copy.tolist()
        #value_copy.remove(value_copy[index_i1][:])
        #value_copy = np.array(value_copy)
    return donnée

def checkDistance(d_cam,dim,cm,L,W,H,name_data):
    longueur = 4.5
    largeur = 1.8
    beta = 28
    #diff = np.abs(d_radar - d_cam)
    for i in range(len(dim)):
        #print('dim',name[i])
        #diff_i = np.abs(d_radar[i]-d_cam[i])
        
        if dim[i] == longueur:
            dimension_new_i = largeur
        else: 
            dimension_new_i = longueur
            
        theta1 = (cm[i,0]-W/2 - L[i]/2)*beta/W
        theta2 = (cm[i,0]-W/2)*beta/W
        dist_i = (dimension_new_i/2)/(np.cos(theta2*np.pi/180)*(np.tan(theta2*np.pi/180)-np.tan(theta1*np.pi/180)))
        #print(dist_i)
        """
        if diff_i > np.abs(dist_i-d_radar[i]):
            #d_cam_new[i] = dist_i
            d_cam_new[i] = distanceModif(cm[i],dimension_new_i,round(L[i]),W,H)
        data_new[:,0] = d_cam_new[:]
        x_new,y_new,z_new = coordonnee_sph(d_cam_new,data[:,2],data[:,4])
        data_new[:,6] = x_new[:]
        data_new[:,8] = y_new[:]
        data_new[:,10] = z_new[:]
        """
    return dist_i
#count = 250
def validation_cam(count):
    name_data = 'data-validation3'
    W = 1280
    H = 720
    if(os.path.isfile('C:/Users/Gauthier_Rotsart/Documents/yolov5/data/images/'+name_data+'/image_'+str(count).zfill(4)+'.jpg') == 1):
        cond = os.path.isfile('C:/Users/Gauthier_Rotsart/Documents/yolov5/runs/detect/'+name_data+'/labels/image_'+str(count).zfill(4)+'.txt') == 1 and os.path.isfile('C:/Users/Gauthier_Rotsart/Documents/yolov5/runs/detect/'+name_data+'/labels/image_'+str(count-1).zfill(4)+'.txt') == 1 and os.path.isfile('C:/Users/Gauthier_Rotsart/Documents/yolov5/runs/detect/'+name_data+'/labels/image_'+str(count-2).zfill(4)+'.txt') == 1
        #print(cond)
        if(cond):
            data0 = pd.read_csv ('C:/Users/Gauthier_Rotsart/Documents/yolov5/runs/detect/'+name_data+'/labels/image_'+str(count-2).zfill(4)+'.txt',sep=' ',header = None).values
            data1 = pd.read_csv ('C:/Users/Gauthier_Rotsart/Documents/yolov5/runs/detect/'+name_data+'/labels/image_'+str(count-1).zfill(4)+'.txt',sep=' ',header = None).values
            data2 = pd.read_csv ('C:/Users/Gauthier_Rotsart/Documents/yolov5/runs/detect/'+name_data+'/labels/image_'+str(count).zfill(4)+'.txt',sep=' ',header = None).values
            #print(data0)
            #denormalisation des datas de yolo
            data0[:,1] = data0[:,1]*W#coord x du centre de masse
            data0[:,2] = data0[:,2]*H#coord y du centre de masse
            data0[:,3] = data0[:,3]*W#largeur du rectangle
            data0[:,4] = data0[:,4]*H#hauteur du rectangle
            
            data1[:,1] = data1[:,1]*W#coord x du centre de masse
            data1[:,2] = data1[:,2]*H#coord y du centre de masse
            data1[:,3] = data1[:,3]*W#largeur du rectangle
            data1[:,4] = data1[:,4]*H#hauteur du rectangle
            
            data2[:,1] = data2[:,1]*W#coord x du centre de masse
            data2[:,2] = data2[:,2]*H#coord y du centre de masse
            data2[:,3] = data2[:,3]*W#largeur du rectangle
            data2[:,4] = data2[:,4]*H#hauteur du rectangle
            
            data0 = isVehicle(data0)
            data1 = isVehicle(data1)
            data2 = isVehicle(data2)
            
            data0 = isTheSame(data0)
            data1 = isTheSame(data1)
            data2 = isTheSame(data2)
            
            #cm0 = np.int32(np.array([data0[:,1].tolist(),data0[:,2].tolist()]).T)
            cm1 = np.int32(np.array([data1[:,1].tolist(),data1[:,2].tolist()]).T)
            cm2 = np.int32(np.array([data2[:,1].tolist(),data2[:,2].tolist()]).T)
            
            #dim1 = dimension(cm0,cm1)
            dim2 = dimension(cm1,cm2)
            
            data = calibration(data2[:,3],cm2,dim2,W,H)
            dist_check = checkDistance(data[:,0], dim2, cm2, data2[:,4], W, H, name_data)
            data_check = data.copy()
            data_check[:,0] = dist_check
            return data, data_check
        else:
            return []