# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:56:45 2021

@author: Gauthier_Rotsart
"""
#Pour mémoire: calibration de la caméra en utilisant les coordonnées homogènes
import numpy as np
import cv2
from calib3d import Point3D, Point2D
import pandas as pd
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
    print('ola1',data1[:,4])
    print('ola',data2[:,4])
    cm1 = np.int32(np.array([data1[:,1].tolist(),data1[:,2].tolist()]).T)
    cm2 = np.int32(np.array([data2[:,1].tolist(),data2[:,2].tolist()]).T)
    #distance parcourue par chaque centre de masse
    distance = np.zeros((len(cm2),2))
    for i in range(len(cm2)):
        value = cm2[i]
        L = np.sqrt((value[0]-cm1[:,0])**2+(value[1]-cm1[:,1])**2).tolist()#association: le plus pres
        distance[i,0] = min(L)
        distance[i,1] = L.index(min(L))
    #changement de coordonnees
    cm1[:,0] = cm1[:,0]-960*np.ones(cm1.shape[0])
    cm1[:,1] = cm1[:,1]-540*np.ones(cm1.shape[0])
    cm2[:,0] = cm2[:,0]-960*np.ones(cm2.shape[0])
    cm2[:,1] = cm2[:,1]-540*np.ones(cm2.shape[0])
    print('cm1',cm1)
    print('cm2',cm2)
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
    return d.tolist(),cm2

def find_intersection(C, d, P, n):
    """
        Finds the intersection between a line and a plane.
        Arguments:
            C - a Point3D of a point on the line
            d - the direction-vector of the line
            P - a Point3D on the plane
            n - the normal vector of the plane
        Returns the Point3D at the intersection between the line and the plane.
    """
    # Write your implementation here
    #j'ai pris simplement une formule d'internet, vu que c'était permis
    point3D = C+d*((np.vdot((P-C),n))/np.vdot(d,n))
    #Point3D(point3D)
    assert isinstance(point3D, Point3D) #and 
    return point3D

def calibration():
    cond = 0
    count = 800#les frames pour la calibration démarrent a 000001500 donc count à 1501
    #count2 = 0
    #count3 = 0
    #count_x = 0
    #count_y = 0
    Xpos = []
    Ypos = []
    Zpos = []
    Xpos2D = []
    Ypos2D = []
    d = []
    #position X,Y,Z de la caméra
    #AMELIORATION: lire le fichier qui donne la position
    Xc = 33402.0
    Yc = -76290.0
    Zc = -53546.0
    while(cond != 1):#lecture de 100 frames de FX
        if os.path.isfile('data/00_'+str(count).zfill(8)+'.csv') == 1:
            data = pd.read_csv('data/00_'+str(count).zfill(8)+'.csv', sep=';').values
            width = len(data)
            #dist = hausdorf(count,count-1)
            for i in range(width):
                if (data[i,6] != -1):#si c'est -1, c'est que le véhicule est cache
                    Xpos.append(data[i,3]-Xc)
                    Ypos.append(data[i,4]-Yc)
                    Zpos.append(data[i,5]-Zc)
                    Xpos2D.append(data[i,6])
                    Ypos2D.append(data[i,7])
                    #d.append(dist[i])
            count+=1
        else:
            cond = 1
    points3D = np.float32([Xpos,Ypos,Zpos])
    points3D = points3D.T
    points2D = np.float32([Xpos2D,Ypos2D])
    points2D = points2D.T
    width = 1920
    height = 1080
    
    camera_matrix = cv2.initCameraMatrix2D([points3D[0:3]],[points2D[0:3]], (width,height))
    _, K, kc, r, t = cv2.calibrateCamera([points3D],[points2D], (width, height), camera_matrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    print(camera_matrix)
    print(K)
    R = cv2.Rodrigues(r[0])[0] #pour avoir une matrice
    T = t[0] #pour avoir un vecteur
    P = K@np.hstack((R,T)) 
    Pinv = np.linalg.pinv(P)
    Kinv = np.linalg.pinv(K)
    rad1, rad2, tan1, tan2, rad3 = kc.flatten()
    
    P = np.array([[650,0,960,0],[0,650,540,0],[0,0,1,0]])
    #print('P3', points3D[-2]+[Xc,Yc,Zc])
    #points = Point2D(P@Point3D(points3D[-2]+[Xc,Yc,Zc]).H)
    #print('P2',points)
    #print('P22',points2D[-2])
    
    return P,R,T
count = 2437#test de la frame 1313
P,R,T = calibration()
Pinv = np.linalg.pinv(P)
n = np.array([[0,0,1]]).T
points_X = []
points_Y = []
points_Z = []
Xc = 33402.0
Yc = -76290.0
Zc = -53546.0
while(os.path.isfile('00_'+str(count).zfill(8)+'.csv') == 1):
    d,cm2 = hausdorf(count-2,count)
    cm2[:,0] = cm2[:,0]+960*np.ones(cm2.shape[0])
    cm2[:,1] = cm2[:,1]+540*np.ones(cm2.shape[0])
    data = pd.read_csv('00_'+str(count).zfill(8)+'.csv', sep=';').values
    point3D_test = data[:,3:6]-[Xc,Yc,Zc]
    #aa = Point3D(point3D_test.T)
    point2D_test = Point2D(P@Point3D(point3D_test.T).H).T
    #print('pixel_estimation',point2D_test)
    #print('pixel_reel_YOLO',cm2)
    for i in range(len(d)):
        points2D = Point2D(cm2[i,0],cm2[i,1])
        X = Point3D(Pinv@points2D.H)
        C = Point3D(-R.T@T)
        points = find_intersection(C,C-X,Point3D(0,0,d[i]*8),n)#d[i]*8#le facteur 10 est la conversion pixel-centimetre
        points_X = np.append(points_X, points[0]+Xc)
        points_Y = np.append(points_Y, points[1]+Yc)
        points_Z = np.append(points_Z, points[2]+Zc)
        #print(Point3D(0,0,d[i]*10))
    count+=1
distance = np.sqrt((points_X-Xc)**2+(points_Y-Yc)**2+(points_Z-Zc)**2)
distance_real = np.sqrt(np.float32((data[:,3]-Xc)**2+(data[:,4]-Yc)**2+(data[:,5]-Zc)**2))
print('distance_estimation',distance/100)
print('distance_reelle',distance_real/100)
"""
données = np.zeros((len(points3D),6))
for i in range(len(points3D)):
    #one_point = Point3D(Xpos[i], Ypos[i],Zpos[i])
    n = np.array([[0,0,1]]).T
    point2D = Point2D(Xpos2D[i], Ypos2D[i])
    
#print("one_point:\n {}\n".format(one_point))
#print("one_point in homogenous coordinates:\n {}\n".format(one_point.H))
    X = Point3D(Pinv@point2D.H)
    C = Point3D(-R.T@T)
    points = find_intersection(C,C-X,Point3D(0,0,points3D[i,2]),n)
    données[i,0] = points3D[i,0]/100
    données[i,1] = points3D[i,1]/100
    données[i,2] = points3D[i,2]/100
    données[i,3] = points.x/100
    données[i,4] = points.y/100
    données[i,5] = points.z/100
#print(points)
"""
    