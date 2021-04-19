# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:09:10 2021

@author: Gauthier_Rotsart
"""
import pandas as pd
import numpy as np
import os.path 
import csv
from main import CreateandSearch
from correction import correction

#Supprime les datas de la caméra qui sont cachées
#Renvoie un nouvel array avec uniquement les datas utiles
def dataCamera(data):
    compteur = 0
    for i in range(len(data)):
        if data[i,6] != -1 and data[i,7] != -1:
            compteur += 1
    data_cam = np.empty((compteur,data.shape[1]), dtype=object)
    compteur = 0
    for i in range(len(data)):
        if data[i,6] != -1 and data[i,7] != -1:
            data_cam[compteur,:] = data[i,:]
            compteur +=1
    return data_cam

#Nouvelle méthode de FX pour les pixels
def dataCamera2(data,W,H):
    compteur = 0
    for i in range(len(data)):
        #print(data[i,6],data[i,7])
        if (data[i,6] >= 0 and data[i,6] <= W) and (data[i,7] >= 0 and data[i,7] <= H):
            compteur +=1
            #print(data[i,6],data[i,7])
    #print(compteur)
    data_cam = np.empty((compteur,data.shape[1]), dtype=object)
    compteur = 0
    for i in range(len(data)):
        if (data[i,6] >= 0 and data[i,6] <= W) and (data[i,7] >= 0 and data[i,7] <= H):
            data_cam[compteur,:] = data[i,:]
            compteur +=1
    return data_cam

#Permet d avoir le même nombre de détection entre la caméra et YOLO
#Cela suppose que si la caméra et YOLO ont capté le même nombre de véhicules,
#ce sont les mêmes  
def association(cm,data,data_YOLO):
    #data_YOLO_new = np.zeros((data_YOLO.shape))
    data_copy = data.copy()
    if len(data) > len(cm):
        data_FX = np.empty((len(cm),data.shape[1]),dtype=object)
        for i in range(len(cm)):
            value = cm[i]
            #L1 = np.sqrt(((value[0]-data[:,6].tolist())**2+(value[1]-data[:,7].tolist())**2)).tolist()
            L1 = np.sqrt(((value[0]-data_copy[:,6].tolist())**2+(value[1]-data_copy[:,7].tolist())**2)).tolist()
            index_i1 = L1.index(min(L1))
            #data_FX[i,:] = data[index_i1,:]
            data_FX[i,:] = data_copy[index_i1,:]
            data_copy = data_copy.tolist()
            data_copy.remove(data_copy[index_i1][:])
            data_copy = np.array(data_copy,dtype=object)
        return cm,data_FX,data_YOLO
    elif len(data) < len(cm):
        cm_x = []
        cm_y = []
        cm_copy = cm.copy()
        data_YOLO_new = np.zeros((len(data),data_YOLO.shape[1]))
        for i in range(len(data)):
            #L1 = np.sqrt(((data[i,6]-cm[:,0])**2+(data[i,7]-cm[:,1])**2)).tolist()
            L1 = np.sqrt(((data[i,6]-np.array(cm_copy[:,0]))**2+(data[i,7]-np.array(cm_copy[:,1]))**2)).tolist()
            index_i1 = L1.index(min(L1))
            #cm_x = np.append(cm_x,cm[index_i1,0])
            #cm_y = np.append(cm_y,cm[index_i1,1])
            cm_x = np.append(cm_x,cm_copy[index_i1,0])
            cm_y = np.append(cm_y,cm_copy[index_i1,1])
            cond = (cm[:,0] == cm_copy[index_i1,0]) & (cm[:,1] == cm_copy[index_i1,1])
            ind = np.where(cond)
            #data_YOLO_new[i,:] = data_YOLO[index_i1,:]
            data_YOLO_new[i,:] = data_YOLO[ind,:]
            cm_copy = cm_copy.tolist()
            cm_copy.remove(cm_copy[index_i1][:])
            cm_copy = np.array(cm_copy)
            cm_new = np.array([cm_x,cm_y]).T
        return np.int32(cm_new),data,data_YOLO_new
    else:
        return cm,data,data_YOLO

#Suppose que le radar et la caméra voient la même chose, pour autant que la 
#distance n'est pas trop grande
def radarCamera(data_radar, data_t, dim, name, cm, data):
    X_true = data_t[:,7]
    Y_true = data_t[:,9]
    Z_true = data_t[:,11]
    d_true = data_t[:,1]
    X_cam = data_t[:,6]
    Y_cam = data_t[:,8]
    Z_cam = data_t[:,10]
    X_radar = np.zeros(len(data_radar))
    Y_radar = np.zeros(len(data_radar))
    Z_radar = np.zeros(len(data_radar))
    d_radar = np.zeros(len(data_radar))
    for i in range(len(data_radar)):
        d_i = data_radar[i,0]
        theta_i = data_radar[i,1]
        phi_i = data_radar[i,2]
        v_i = data_radar[i,3]
        X_radar[i],Y_radar[i],Z_radar[i] = coordonnee_sph(d_i,theta_i,phi_i)
        d_radar[i] = d_i
    if len(data_radar) > len(data_t):#il faut supprimer des datas radars
        data_r_new = np.zeros((len(data_t),4))
        d_radar_copy = d_radar.copy()
        for i in range(len(data_t)):
            #L1 = np.sqrt(((X_true[i]-X_radar.tolist())**2+(Y_true[i]-Y_radar.tolist())**2)+(Z_true[i]-Z_radar.tolist())**2).tolist()
            d_true_i = d_true[i]
            #L1 = np.sqrt(((d_true_i-d_radar.tolist())**2)).tolist()
            L1 = np.sqrt(((d_true_i-d_radar_copy.tolist())**2)).tolist()
            index_i1 = L1.index(min(L1))
            #data_r_new[i,:] = data_radar[index_i1,:]
            ind = np.where((data_radar[:,0] == d_radar_copy[index_i1]))
            #print(ind)
            #print(data_radar[:,0])
            #print(d_radar_copy[index_i1])
            data_r_new[i,:] = data_radar[ind,:]
            d_radar_copy = d_radar_copy.tolist()
            d_radar_copy.remove(d_radar_copy[index_i1])
            d_radar_copy = np.array(d_radar_copy)
        return data_t, data_r_new, dim, name, cm, data
    elif len(data_radar) < len(data_t):#il faut supprimer des datas cameras
        data_t_new = np.zeros((len(data_radar),12))
        dim_new = np.zeros(len(data_radar))
        name_new = np.empty(len(data_radar),dtype=str)
        cm_new = np.zeros((len(data_radar),cm.shape[1]))
        data_new = np.zeros((len(data_radar),data.shape[1]))
        d_true_copy = d_true.copy()
        for i in range(len(data_radar)):
            #L1 = np.sqrt(((X_radar[i]-X_true.tolist())**2+(Y_radar[i]-Y_true.tolist())**2)+(Z_radar[i]-Z_true.tolist())**2).tolist()
            #L1 = np.sqrt(((d_radar[i]-d_true.tolist())**2)).tolist()
            L1 = np.sqrt(((d_radar[i]-d_true_copy.tolist())**2)).tolist()
            #print(L1)
            #print(d_radar[i])
            index_i1 = L1.index(min(L1))
            ind = np.where((data_t[:,1] == d_true_copy[index_i1]))
            #data_t_new[i,:] = data_t[index_i1,:]
            #print(ind)
            #print(name[ind][0])
            #print(name_new)
            data_t_new[i,:] = data_t[ind,:]
            dim_new[i] = dim[ind]
            name_new[i] = name[ind][0]
            cm_new[i,:] = cm[ind,:]
            data_new[i,:] = data[ind,:]
            d_true_copy = d_true_copy.tolist()
            d_true_copy.remove(d_true_copy[index_i1])
            d_true_copy = np.array(d_true_copy)
            
        #print(data_t_new)
        return data_t_new, data_radar,dim_new,name_new,cm_new,data_new
    else:
        data_r_new = np.zeros((len(data_t),4))
        data_radar_copy = data_radar.copy()
        for i in range(len(data_t)):
            #L1 = np.sqrt(((data_t[i,0]-data_radar[:,0].tolist())**2)).tolist()
            L1 = np.sqrt(((data_t[i,0]-data_radar_copy[:,0].tolist())**2)).tolist()
            index_i1 = L1.index(min(L1))
            ind = np.where((data_radar[:,0] == data_radar_copy[index_i1,0]))
            #data_r_new[i,:] = data_radar[index_i1,:]
            data_r_new[i,:] = data_radar[ind,:]
            data_radar_copy = data_radar_copy.tolist()
            data_radar_copy.remove(data_radar_copy[index_i1][:])
            data_radar_copy = np.array(data_radar_copy)
        return data_t,data_r_new,dim,name,cm,data
    

#Conversion coordonnées sphériques vers coordonnées cartésiennes
def coordonnee_sph(r,theta,phi):
    theta = np.radians(theta)
    phi = np.radians(phi)
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z

#Détermine si la voiture est vue de dos ou de côté
#Retourne la largeur en metre correspondante
def dimension(cm1,cm2,data,data_YOLO):
    vehicule_dim = pd.read_csv('C:/Users/Gauthier_Rotsart/Downloads/vehicule_dimension.csv',sep=';').values
    longueur = vehicule_dim[:,1]
    largeur = vehicule_dim[:,2]
    dimension = np.zeros(len(cm2))
    name = []
    for i in range(len(cm2)):
        value = cm2[i]
        L1 = np.sqrt(((value[0]-data[:,6].tolist())**2+(value[1]-data[:,7].tolist())**2)).tolist()
        index_i1 = L1.index(min(L1))
        name_i = data[index_i1,1]
        L2 = np.sqrt((value[0]-cm1[:,0])**2+(value[1]-cm1[:,1])**2).tolist()
        index_i2 = L2.index(min(L2))
        delta_x = np.abs(cm1[index_i2,0]-cm2[i,0])
        delta_y = np.abs(cm1[index_i2,1]-cm2[i,1])
        t = np.char.split(name_i,sep='_').tolist()
        t = t[0:-1]#supprime le numero de l instance du vehicule
        name_i = t[0]+'_'
        for j in range(len(t)-1):
            if j+1 == len(t)-1:
                name_i += t[j+1]
            else:
                name_i += t[j+1]+'_'
        name = np.append(name,name_i)
        ind = np.where(vehicule_dim[:,0] == name_i)
        #IL FAUT PRENDRE EN COMPTE LES NOUVELLES DIMENSIONS !!
        """
        if data_YOLO[i,3] - data_YOLO[i,4] >= 0:#vue de côté
            if len(ind[0])==0:
                dimension[i] = 7.6
            else:
                #print(delta_x-delta_y)
                dimension[i] = longueur[ind]/100
        else:
            if len(ind[0])==0:
                dimension[i] = 3.12
            else:
                #print(delta_y-delta_x)
                dimension[i] = largeur[ind]/100
            
        
        """
        if delta_x > delta_y:#avance principalement verticalement dans l'image
            if len(ind[0])==0:
                dimension[i] = 7.6
            else:
                #print(delta_x-delta_y)
                dimension[i] = longueur[ind]/100
        elif delta_x < delta_y:#avance principalement horizontalement dans l'image
            if len(ind[0])==0:
                dimension[i] = 3.12
            else:
                #print(delta_y-delta_x)
                dimension[i] = largeur[ind]/100
        else:#n'avance pas
            if(cm2[i,0] > 1160):
                if len(ind[0])==0:
                    dimension[i] = 7.6
                else:
                    dimension[i] = longueur[ind]/100
            else:
                if len(ind[0])==0:
                    dimension[i] = 3.12
                else:
                    dimension[i] = largeur[ind]/100
        
    return dimension,name

#Estimation de la position d'un véhicule dans le repère de la caméra
#Retourne les coordonnées sphériques et cartésiennes
def calibration(L,cm,dimension,data,W,H):
    beta = 28
    distance = np.zeros(len(cm))
    distance_real = np.zeros(len(cm))
    theta = np.zeros(len(cm))
    theta_real = np.zeros(len(cm))
    phi = np.zeros(len(cm))
    phi_real = np.zeros(len(cm))
    x = np.zeros(len(cm))
    y = np.zeros(len(cm))
    z = np.zeros(len(cm))
    donnée = np.zeros((len(cm),12))
    data_copy = data.copy()
    for i in range(len(cm)):
        #theta1 = (H[i]/2+(800-cm[i,0]))*beta/1920
        #theta2 = ((800-cm[i,0]))*beta/1920
        #theta1 = (-H[i]/2-(960-cm[i,0]))*28/1920
        #theta2 = (H[i]/2-(960-cm[i,0]))*28/1920
        #distance[i] = dimension[i]/(np.tan(theta2*np.pi/180)-np.tan((theta1)*np.pi/180)) 
        #if dimension[i] < 4:
            #print('old',distance[i])
            #distance[i] += 2.5
            #print('new',distance[i])
        #distance[i] = np.sqrt(distance[i]**2+5.57**2) 
        #print(i)
        #print(cm[i])
        value = cm[i]
        #L1 = np.sqrt(((value[0]-data[:,6].tolist())**2+(value[1]-data[:,7].tolist())**2)).tolist()
        L1 = np.sqrt(((value[0]-data_copy[:,6].tolist())**2+(value[1]-data_copy[:,7].tolist())**2)).tolist()
        index_i1 = L1.index(min(L1))
        cond = (value[0] == data_copy[index_i1,6]) & (value[1] == data_copy[index_i1,7])
        ind = np.where(cond)

        #theta1 = (-H[i]/2-(960-cm[i,0]))*28/1920
        #theta2 = (H[i]/2-(960-cm[i,0]))*28/1920
        theta1 = (cm[i,0]-W/2 - L[i]/2)*beta/W
        theta2 = (cm[i,0]-W/2)*beta/W
        a = ((cm[i,0]-L[i]/2)-W/2)*(dimension[i]/(L[i]))
        #print('hehe',a)
        #distance[i] = (dimension[i]/2+a)/np.sin((theta2)*np.pi/180)
        #print(distance[i])
        #distance[i] = np.sqrt(distance[i]**2 + 5.57**2)
        distance[i] = (dimension[i]/2)/(np.cos(theta2*np.pi/180)*(np.tan(theta2*np.pi/180)-np.tan(theta1*np.pi/180)))
        dd = distanceModif(cm[i],dimension[i],round(L[i]),W)
        #print('old distance', distance[i])
        print('new distance',np.mean(dd))
        distance[i] = dd
        #distance_real[i] = np.sqrt((data[index_i1,3]-Xc)**2+(data[index_i1,4]-Yc)**2+(data[index_i1,5]-Zc)**2)/100
        distance_real[i] = np.sqrt((data_copy[index_i1,3]-Xc)**2+(data_copy[index_i1,4]-Yc)**2+(data_copy[index_i1,5]-Zc)**2)/100
        #print('real distance', distance_real[i])
        donnée[i,0] = distance[i]
        donnée[i,1] = distance_real[i]
        theta[i] = 90 - pitch + (cm[i,1]-H/2)*beta/W
        #print(index_i1)
        #theta_real[i] = np.arccos((data[index_i1,5]-Zc)/(distance_real[i]*100))*180/np.pi
        theta_real[i] = np.arccos((data_copy[index_i1,5]-Zc)/(distance_real[i]*100))*180/np.pi
        phi[i] = yaw + (cm[i,0]-W/2)*beta/W
        #phi_real[i] = np.arctan2((data[index_i1,4]-Yc),(data[index_i1,3]-Xc))*180/np.pi
        phi_real[i] = np.arctan2((data_copy[index_i1,4]-Yc),(data_copy[index_i1,3]-Xc))*180/np.pi
        donnée[i,2] = theta[i]
        donnée[i,3] = theta_real[i]
        donnée[i,4] = phi[i]
        donnée[i,5] = phi_real[i]
        x[i],y[i],z[i] = coordonnee_sph(distance[i],theta[i],phi[i])
        donnée[i,6] = x[i]
        #donnée[i,7] = (data[index_i1,3]-Xc)/100
        donnée[i,7] = (data_copy[index_i1,3]-Xc)/100
        donnée[i,8] = y[i]
        #donnée[i,9] = (data[index_i1,4]-Yc)/100
        donnée[i,9] = (data_copy[index_i1,4]-Yc)/100
        donnée[i,10] = z[i]
        #donnée[i,11] = (data[index_i1,5]-Zc)/100
        donnée[i,11] = (data_copy[index_i1,5]-Zc)/100
        #print('distance', distance[i],cm[i])
        data_copy = data_copy.tolist()
        data_copy.remove(data_copy[index_i1][:])
        data_copy = np.array(data_copy,dtype=object)
    return donnée

def distanceModif(cc,dim,L,W):
    beta = 28
    cc[0] = cc[0] - W/2
    theta1 = (cc[0]- L/2)*beta/W
    #theta2 = (cm-W/2)*beta/W
    #xi = cc[0] +L/2 
    #theta2 = xi*beta/W
    distance = np.zeros(L//2)#(dim)/(np.cos(theta2*np.pi/180)*(np.tan(theta2*np.pi/180)-np.tan(theta1*np.pi/180)))
    
    for i in range((L//2)):
        xi = cc[0] - L/2 + (2*i+1)
        theta2 = xi*beta/W
        N = (i+1)/(L//2)
        distance[i] = (dim*N)/(np.cos(theta2*np.pi/180)*(np.tan(theta2*np.pi/180)-np.tan(theta1*np.pi/180)))
    std = np.std(distance)
    #print(distance)
    mean = np.mean(distance)
    count = 0
    count1 = 0
    for i in range(len(distance)):
        if (distance[i] < mean+std and distance[i] > mean-std):
            count+=distance[i]
            count1+=1
    distance_moy = count/count1
    return distance_moy
#Estime la vitesse d'un véhicule en prenant deux frames
#Ici, c'est la frame t et t-2
def vitesse(data_t,data_old,cm1,cm2,largeur,hauteur):
    vitesse = np.zeros((len(data_t),2))
    dt = 10/30
    print('cm1',cm1)
    print('cm2',cm2)
    for i in range(len(cm2)):
        #value = np.array([data_t[i,6],data_t[i,8],data_t[i,10]]).T
        #L1 = np.sqrt(((value[0]-data_old[:,6].tolist())**2+(value[1]-data_old[:,8].tolist())**2+(value[2]-data_old[:,10].tolist())**2)).tolist()
        value = cm2[i]
        L1 = np.sqrt(((value[0]-cm1[:,0].tolist())**2+(value[1]-cm1[:,1].tolist())**2)).tolist()
        index_i1 = L1.index(min(L1))
        value2 = cm2[i]
        L2 = np.sqrt(((value2[0]-data[:,6].tolist())**2+(value2[1]-data[:,7].tolist())**2)).tolist()
        index_i2 = L2.index(min(L2))
        print('association',cm2[i],cm1[index_i1])
        #value3 = cm2[i]
        #L = np.sqrt((value3[0]-cm1[:,0])**2+(value3[1]-cm1[:,1])**2).tolist()#association: le plus pres
        #index_i3 = L.index(min(L))
        #distance = value3[0] - cm1[index_i3,0]
        #v2[i] = (distance/dt)
        #print(cm1[index_i1,:],cm2[i,:])
        #print(distance)
        #if data2[i,0]==2:
        x1 = data_old[index_i1,6]
        y1 = data_old[index_i1,8]
        z1 = data_old[index_i1,10]
        print('data_old', data_old[index_i1,6],data_old[index_i1,8],data_old[index_i1,10])
        x2 = data_t[i,6]
        y2 = data_t[i,8]
        z2 = data_t[i,10]
        print('data t',data_t[i,6],data_t[i,8],data_t[i,10])
        distance_parc = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        print('distance parcourue', distance_parc)
        vitesse[i,0] = (np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)/dt)
        vitesse[i,1] = (data[index_i2,8]/100)
        print('vitesse estimée', vitesse[i,0])
        print('vitesse réelle', vitesse[i,1])
    return vitesse

def isVehicle(data):
    compteur = 0
    for i in range(len(data)):
        if data[i,0] != 0 and data[i,0] != 9:
            compteur+=1
    data_new = np.empty((compteur,data.shape[1]))
    compteur = 0
    for i in range(len(data)):
        if data[i,0] != 0 and data[i,0] != 9:
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
def checkDistance(d_radar,d_cam,dim,name,cm,L,W,H):
    vehicule_dim = pd.read_csv('C:/Users/Gauthier_Rotsart/Downloads/vehicule_dimension.csv',sep=';').values
    longueur = vehicule_dim[:,1]
    largeur = vehicule_dim[:,2]
    beta = 28
    #diff = np.abs(d_radar - d_cam)
    d_cam_new = d_cam.copy()
    for i in range(len(dim)):
        diff_i = np.abs(d_radar[i]-d_cam[i])
        #if diff_i > 20:
        ind = np.where(vehicule_dim[:,0] == name[i])
        longueur_i = longueur[ind]/100
        largeur_i = largeur[ind]/100
        if len(ind[0])==0:
            if dim[i] == 7.6:
                dimension_new_i = 3.12
            else:
                dimension_new_i = 7.6
        else:
            if dim[i] == longueur_i:
                dimension_new_i = largeur_i
            else: 
                dimension_new_i = longueur_i
         
        theta1 = (cm[i,0]-W/2 - L[i]/2)*beta/W
        theta2 = (cm[i,0]-W/2)*beta/W
        dist_i = (dimension_new_i/2)/(np.cos(theta2*np.pi/180)*(np.tan(theta2*np.pi/180)-np.tan(theta1*np.pi/180)))
        if diff_i > np.abs(dist_i-d_radar[i]):
            print('changed')
            #d_cam_new[i] = dist_i
            d_cam_new[i] = distanceModif(cm[i],dimension_new_i,round(L[i]),W)
    return d_cam_new

def aire(data1,data2):
    L1 = data1[:,3]
    H1 = data1[:,4]
    L2 = data2[:,3]
    H2 = data2[:,4]
    aire = np.zeros(len(H2))
    for i in range(len(L2)):
        L = np.sqrt(((L2[i]-L1[:].tolist())**2+(H2[i]-H1[:].tolist())**2)).tolist()
        index_i1 = L.index(min(L))
        aire[i] = (L2[i]*H2[i]-L1[index_i1]*H1[index_i1])/(L1[index_i1]*H1[index_i1])
    return aire*100
        
#Fonction principale qui lance toutes les sous fonctions
count = 1150#test de la frame 2437
points_X = []
points_Y = []
points_Z = []
Xc = 33442.0
Yc = -76290.0
Zc = -53546.0
pitch = -12.999
yaw = -2.5
W = 1920
H = 1280#1080
compteur = 0
ordre = []
csv_folder= 'C:/Users/Gauthier_Rotsart/Documents/yolov5/data/images/data_jour/cam_00/Excel'#'/home/kdesousa/Documents/GitHub/TFE/Kalman/2021_04_06_15_40_39_604/cam_00'
pos_cam = os.path.join(csv_folder,'pos_cam_00.csv')
with open('data_est.csv','w',newline='') as myWriter, open('data.csv','w',newline='') as myWriter2:
    writer = csv.writer(myWriter,delimiter=';')
    writer2 = csv.writer(myWriter2,delimiter=';')
    while(os.path.isfile('C:/Users/Gauthier_Rotsart/Documents/yolov5/data/images/data_jour/cam_00/Excel/00_'+str(count).zfill(8)+'.csv') == 1):
        data_ant = pd.read_csv('C:/Users/Gauthier_Rotsart/Documents/yolov5/data/images/data_jour/cam_00/Excel/00_'+str(count-10).zfill(8)+'.csv', sep=';').values
        data = pd.read_csv('C:/Users/Gauthier_Rotsart/Documents/yolov5/data/images/data_jour/cam_00/Excel/00_'+str(count).zfill(8)+'.csv', sep=';').values
        data0 = pd.read_csv ('C:/Users/Gauthier_Rotsart/Documents/yolov5/runs/detect/exp6/labels/00_'+str(count-20).zfill(8)+'.txt',sep=' ',header = None).values
        data1 = pd.read_csv ('C:/Users/Gauthier_Rotsart/Documents/yolov5/runs/detect/exp6/labels/00_'+str(count-10).zfill(8)+'.txt',sep=' ',header = None).values
        data2 = pd.read_csv ('C:/Users/Gauthier_Rotsart/Documents/yolov5/runs/detect/exp6/labels/00_'+str(count).zfill(8)+'.txt',sep=' ',header = None).values
        
        #data_ant = dataCamera(data_ant)
        #data = dataCamera(data)
        data_ant = dataCamera2(data_ant,W,H)
        data_test = data
        data = dataCamera2(data,W,H)
        if(data.shape != (0,60) and data_ant.shape != (0,60)):
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
            
            cm0 = np.int32(np.array([data0[:,1].tolist(),data0[:,2].tolist()]).T)
            cm1 = np.int32(np.array([data1[:,1].tolist(),data1[:,2].tolist()]).T)
            cm2 = np.int32(np.array([data2[:,1].tolist(),data2[:,2].tolist()]).T)
    
            #print('pixel_reel_YOLO')
            
            cm1,data_ant,data1 = association(cm1,data_ant,data1)
            cm2,data,data2 = association(cm2,data,data2)
            area = aire(data1,data2)
            print(area)
            dim1,name1 = dimension(cm0,cm1,data_ant,data1)
            dim2,name2 = dimension(cm1,cm2,data,data2)
            #print(cm2)
            data_old = calibration(data1[:,3],cm1,dim1,data_ant,W,H)
            data_t = calibration(data2[:,3],cm2,dim2,data,W,H)
            
            csv_folder= 'C:/Users/Gauthier_Rotsart/Documents/yolov5/data/images/data_jour/cam_00/Excel'
            pos_cam = os.path.join(csv_folder,'pos_cam_00.csv')
            df = pd.read_csv(pos_cam, sep =';')
            pos_cam = df.values[1,:]#[df.iloc[2]['Xpos'],df.iloc[2]['Ypos'],df.iloc[2]['Zpos']]
            data_r = CreateandSearch('C:/Users/Gauthier_Rotsart/Documents/yolov5/data/images/data_jour/cam_00/Excel/00_'+str(count).zfill(8)+'.csv', pos_cam)
            data_r_test = data_r
            for i in range(len(data_r)):
                data_r[i,0] = data_r[i,0] + np.random.rand()
            data_t, data_r,dim2,name2,cm2,data2 = radarCamera(data_r,data_t,dim2,name2,cm2,data2)
            d_cam = checkDistance(data_r[:,0], data_t[:,0], dim2, name2, cm2, data2[:,3], W, H)
            #FAIRE L UPDATE DIRECTEMENT DANS CHECKDISTANCE
            data_t[:,0] = d_cam
            x_new,y_new,z_new = coordonnee_sph(d_cam,data_t[:,2],data_t[:,4])
            data_t[:,6] = x_new
            data_t[:,8] = y_new
            data_t[:,10] = z_new
            v = vitesse(data_t,data_old,cm1,cm2,data2[:,3],data2[:,4])
            #ecriture du fichier csv 
            for i in range(len(data_t)):
                d_i1 = d_cam[i]#data_t[i,0]
                theta_i1 = data_t[i,2]
                phi_i1 = data_t[i,4]
                v_i1 = v[i,0]
                d_ri1 = data_r[i,0]
                theta_ri1 = data_r[i,1]#*180/np.pi
                phi_ri1 = data_r[i,2]#*180/np.pi
                v_ri1 = data_r[i,3]
                vect_i = np.array([d_i1,theta_i1,phi_i1,v_i1,d_ri1,theta_ri1,phi_ri1,v_ri1])
                data_corr = correction(vect_i)
                #print(data_corr)
                d_i2 = data_t[i,1]
                theta_i2 = data_t[i,3]
                phi_i2 = data_t[i,5]
                v_i2 = v[i,1]
                #writer.writerow((count,d_i1,theta_i1,phi_i1,v_i1))#Pour connaitre la frame
                #writer2.writerow((count,d_i2,theta_i2,phi_i2,v_i2))
                #writer.writerow((d_i1,theta_i1,phi_i1,v_i1,d_ri1,theta_ri1,phi_ri1,v_ri1))
                writer.writerow((count,data_corr[0],data_corr[1],data_corr[2],data_corr[3],data_corr[4],data_corr[5],data_corr[6],data_corr[7]))
                writer2.writerow((count,d_i2,theta_i2,phi_i2,v_i2))
                ordre = np.append(ordre,count)
        if compteur == 100:
            print('100 frames faites', count)
            compteur = 0
        compteur +=1
        count+=10000
A = pd.read_csv('data.csv',sep=';',header = None)
B = pd.read_csv('data_est.csv',sep=';',header = None)
    
    
    
