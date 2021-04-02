# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:09:10 2021

@author: Gauthier_Rotsart
"""
import pandas as pd
import numpy as np
import os.path 
import math

def coordonnee_sph(r,theta,phi):
    theta = np.radians(theta)
    phi = np.radians(phi)
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z
def dimension(cm1,cm2,data):
    #détermine les dimensions de la voiture
    vehicule_dim = pd.read_csv('vehicule_dimension.csv',sep=';').values
    longueur = vehicule_dim[:,1]
    largeur = vehicule_dim[:,2]
    dimension = np.zeros(len(cm2))
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
        
        ind = np.where(vehicule_dim[:,0] == name_i)
        if delta_x > delta_y:
            dimension[i] = longueur[ind]/100
        elif delta_x < delta_y:
            dimension[i] = largeur[ind]/100
        else:
            if(cm2[i,0] > 1160):
                dimension[i] = longueur[ind]/100
            else:
                dimension[i] = largeur[ind]/100
    return dimension

def calibration(H,cm,dimension,data):
    beta = 28
    distance = np.zeros(len(H))
    distance_real = np.zeros(len(H))
    theta = np.zeros(len(H))
    theta_real = np.zeros(len(H))
    phi = np.zeros(len(H))
    phi_real = np.zeros(len(H))
    x = np.zeros(len(H))
    y = np.zeros(len(H))
    z = np.zeros(len(H))
    donnée = np.zeros((len(H),12))
    """
    donnée[0,0] = 'distance estimée'
    donnée[1,0] = 'distance reelle'
    donnée[2,0] = 'theta estimé'
    donnée[3,0] = 'theta reel'
    donnée[4,0] = 'phi estimé'
    donnée[5,0] = 'phi reel'
    """
    for i in range(len(H)):
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
        value = cm[i]
        L1 = np.sqrt(((value[0]-data[:,6].tolist())**2+(value[1]-data[:,7].tolist())**2)).tolist()
        index_i1 = L1.index(min(L1))
        theta1 = (-H[i]/2-(960-cm[i,0]))*28/1920
        theta2 = (H[i]/2-(960-cm[i,0]))*28/1920
        #theta2 = ((cm[i,0]-H[i]/2)-960)*28/1920
        #theta2 = (960-cm[i,0])*28/1920
        #alpha = (H[i]+(960-cm[i,0]))*28/1920
        a = ((cm[i,0]-H[i]/2)-960)*(dimension[i]/(H[i]))
        #print('hehe',a)
        distance[i] = (dimension[i]+a)/np.sin((theta2)*np.pi/180)
        distance[i] = np.sqrt(distance[i]**2 + 5.57**2)
        #distance[i] = dimension[i]/(np.tan(theta2*np.pi/180)-np.tan((theta1)*np.pi/180))
        #distance[i] = (dimension[i])/(np.sin(theta2*np.pi/180)-np.sin((theta1)*np.pi/180))
        distance_real[i] = np.sqrt((data[index_i1,3]-Xc)**2+(data[index_i1,4]-Yc)**2+(data[index_i1,5]-Zc)**2)/100
        donnée[i,0] = distance[i]
        donnée[i,1] = distance_real[i]
        theta[i] = 90 - pitch + (cm[i,1]-540)*28/1920
        #print(index_i1)
        theta_real[i] = np.arccos((data[index_i1,5]-Zc)/(distance_real[i]*100))*180/np.pi
        phi[i] = yaw + (cm[i,0]-960)*28/1920
        phi_real[i] = np.arctan2((data[index_i1,4]-Yc),(data[index_i1,3]-Xc))*180/np.pi
        donnée[i,2] = theta[i]
        donnée[i,3] = theta_real[i]
        donnée[i,4] = phi[i]
        donnée[i,5] = phi_real[i]
        x[i],y[i],z[i] = coordonnee_sph(distance[i],theta[i],phi[i])
        donnée[i,6] = x[i]
        donnée[i,7] = (data[index_i1,3]-Xc)/100
        donnée[i,8] = y[i]
        donnée[i,9] = (data[index_i1,4]-Yc)/100
        donnée[i,10] = z[i]
        donnée[i,11] = (data[index_i1,5]-Zc)/100
        print('distance', distance[i],cm[i])
    return donnée
def vitesse(data_t,data_old,cm1,cm2,largeur,hauteur):
    vitesse = np.zeros((len(data_t),2))
    dt = 2/30
    #extraire les coordonnees x,y,z de data_t et data_old et en prendre la distance puis v = d/t
    for i in range(len(data_t)):
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
        pitch = np.radians(-12.999)
        yaw = np.radians(-2.5)
        #Rz = np.array([[np.cos(-yaw),-np.sin(-yaw),0],[np.sin(-yaw),np.cos(-yaw),0],[0,0,1]])
        #Ry = np.array([[np.cos(-pitch),0,np.sin(-pitch)],[0,1,0],[-np.sin(-pitch),0,np.cos(-pitch)]])
        #R = Rz@Ry
        #x1,y1,z1 = R@np.array([data_old[index_i1,6],data_old[index_i1,8],data_old[index_i1,10]]).T
        x1 = data_old[index_i1,6]
        y1 = data_old[index_i1,8]
        z1 = data_old[index_i1,10]
        #x2,y2,z2 = R@np.array([data_t[i,6],data_t[i,8],data_t[i,10]]).T
        #print(x2,y2,z2)
        x2 = data_t[i,6]
        y2 = data_t[i,8]
        z2 = data_t[i,10]
        print(x1,y1,z1,x2,y2,z2)
        vitesse[i,0] = (np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)/dt)
        #print(np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2))
        vitesse[i,1] = (data[index_i2,8]/100)
    return vitesse

count = 2437#2000#test de la frame 2437
points_X = []
points_Y = []
points_Z = []
Xc = 33442.0
Yc = -76290.0
Zc = -53546.0
pitch = -12.999
yaw = -2.5
W = 1920
H = 1080
vit1 = []
vit2 = []
while(os.path.isfile('00_'+str(count).zfill(8)+'.csv') == 1):
    data_ant = pd.read_csv('00_'+str(count-2).zfill(8)+'.csv', sep=';').values
    data = pd.read_csv('00_'+str(count).zfill(8)+'.csv', sep=';').values
    data0 = pd.read_csv ('00_'+str(count-4).zfill(8)+'.txt',sep=' ',header = None).values
    data1 = pd.read_csv ('00_'+str(count-2).zfill(8)+'.txt',sep=' ',header = None).values
    data2 = pd.read_csv ('00_'+str(count).zfill(8)+'.txt',sep=' ',header = None).values
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
    
    cm0 = np.int32(np.array([data0[:,1].tolist(),data0[:,2].tolist()]).T)
    cm1 = np.int32(np.array([data1[:,1].tolist(),data1[:,2].tolist()]).T)
    cm2 = np.int32(np.array([data2[:,1].tolist(),data2[:,2].tolist()]).T)
    
    dim1 = dimension(cm0,cm1,data_ant)
    dim2 = dimension(cm1,cm2,data)
    point3D_test = data[:,3:6]-[Xc,Yc,Zc]
    print('pixel_reel_YOLO')
    print(cm2)
    data_old = calibration(data1[:,3],cm1,dim1,data_ant)
    data_t = calibration(data2[:,3],cm2,dim2,data)
    v2 = np.zeros(len(data_t))
    dt = 2/30
    #extraire les coordonnees x,y,z de data_t et data_old et en prendre la distance puis v = d/t
    #for i in range(len):
    v = vitesse(data_t,data_old,cm1,cm2,data2[:,3],data2[:,4])
    for i in range(len(cm2)):
        if data2[i,0] == 2:
            vit1 = np.append(vit1,v[i,0])
            vit2 = np.append(vit2,v[i,1])
        else:
            print('pas une voiture')
    diff = np.abs(vit1-vit2)
    cc = 0
    cc2 = 0
    s = np.std(diff)
    for i in range(len(diff)):
        if diff[i] < s:
            cc += diff[i]
            cc2 += 1
    print('moyenne',cc/cc2)
    d = np.zeros(len(cm2))
    for i in range(len(cm2)):
        ui = cm2[i,0] - round(data2[i,3]//2)
        vi = cm2[i,1] - round(data2[i,4]//2)
        distance = []
        #for j in range(round(data2[i,3])):
            #ui += j
            #print(ui + j)
            #vi = cm2[i,1] - round(data2[i,4]//2)
        for k in range(round(data2[i,3])):
            theta2 = (data2[i,3]/2-(960-(ui+k)))*28/1920
            #theta2 = ((cm[i,0]-H[i]/2)-960)*28/1920
            #theta2 = (960-cm[i,0])*28/1920
            #alpha = (H[i]+(960-cm[i,0]))*28/1920
            a = ((ui+k-data2[i,3]/2)-960)*(dim2[i]/(data2[i,3]))
            #print('hehe',a)
            distance = np.append(distance,(dim2[i]+a)/np.sin((theta2)*np.pi/180))
        #print(vi)
        d[i] = np.mean(distance)
        print('cc')
    
    count+=1
"""
diff = np.abs(vit1-vit2)
s= np.std(diff)
cc = 0
cc2 = 0
for i in range(len(diff)):
    if diff[i]<s:
        cc+=diff[i]
        cc2+=1
print('moyenn', cc/cc2)
c = 0
d = np.zeros(len(cm2))
#h = (5.57**2)*np.ones(len(H)).T
"""


