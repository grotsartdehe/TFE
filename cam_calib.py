# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:20:51 2021

@author: Gauthier_Rotsart
"""
#https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

import numpy as np
import cv2
from calib3d import Point3D, Point2D
import pandas as pd
data1 = pd.read_csv("01_00003181.csv", sep=';').values
data2 = pd.read_csv("01_00003542.csv", sep=';').values
Xpos1 = np.zeros(12)
Ypos1 = np.zeros(12)
Zpos1 = np.zeros(12)
Xpos2D1 = np.zeros(12)
Ypos2D1 = np.zeros(12)

Xpos2 = np.zeros(16)
Ypos2 = np.zeros(16)
Zpos2 = np.zeros(16)
Xpos2D2 = np.zeros(16)
Ypos2D2 = np.zeros(16)
count = 0
for i in range(17):
    if (data1[i,6] != -1):
        j = i - count
        Xpos1[j] = data1[i,3]
        Ypos1[j] = data1[i,4]
        Zpos1[j] = data1[i,5]
        Xpos2D1[j] = data1[i,6]
        Ypos2D1[j] = data1[i,7]
    else:
        print('error1')
        count = count + 1
count = 0
for i in range(22):
    if (data2[i,6] != -1):
        j = i - count
        Xpos2[j] = data2[i,3]
        Ypos2[j] = data2[i,4]
        Zpos2[j] = data2[i,5]
        Xpos2D2[j] = data2[i,6]
        Ypos2D2[j] = data2[i,7]
    else:
        print('error2')
        count = count + 1
Xpos = np.append(Xpos1,Xpos2)
Ypos = np.append(Ypos1,Ypos2)
Zpos = np.append(Zpos1,Zpos2)
Xpos2D = np.append(Xpos2D1,Xpos2D2)
Ypos2D = np.append(Ypos2D1,Ypos2D2)
points3D = np.float32([Xpos,Ypos,Zpos])
points3D = points3D.T
points2D = np.float32([Xpos2D,Ypos2D])
points2D = points2D.T
#Xpos = data[:,3]
#Ypos = data[:,4]
#Zpos = data[:,5]
#Xpos2D = data[:,6]
#Ypos2D = data[:,7]
def center(left, right, top, bot):
    #les rectangles sont tels que le coin superieur gauche est (left, top) et le coin
    #inferieur droit est (right, bot)
    #la fonction retourne le centre de masse des N vehicules detectes par YOLO
    N = len(left)#nombre de cibles detectees par YOLO
    cm = np.zeros((N,2))#tableau de N lignes avec 2 colonnes (x,y)
    for i in range(N):
        cm[i,0] = (left+right)//2
        cm[i,1] = (bot+top)//2
    return cm
    
    
#Prendre 10 frames de FX et en retirer les coordonnées (x,y,z) du centre de masse

#Appliquer YOLO sur les 10 frames et déduire les coordonnées du centre de masse 


#points3D = np.float32([[1,11,21], [2,12,22], [3,13,23], [4,14,24], [5,15,25], [6,16,26]])
#points2D = np.float32([[1.5, 11], [2.3,12.3], [3,13.12],[4.05, 14.54],[5.2,15.3], [6.5, 16.7]])
#height, width = image.shape
width = 1920
height = 1080
camera_matrix = cv2.initCameraMatrix2D([points3D],[points2D], (width,height))
_, K, kc, r, t = cv2.calibrateCamera([points3D],[points2D], (width, height), camera_matrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
R = cv2.Rodrigues(r[0])[0] #pour avoir une matrice
T = t[0] #pour avoir un vecteur
print("Camera matrix:\n", K)
print("distorsion:\n", kc)
print("rotation vector:\n", r)
print("translation vector:\n", t)
one_point = Point3D(Xpos[0], Ypos[0],Zpos[0])
print("one_point:\n {}\n".format(one_point))
print("one_point in homogenous coordinates:\n {}\n".format(one_point.H))
P = K@np.hstack((R,T)) 
points = Point2D(P@one_point.H)
print(points)



