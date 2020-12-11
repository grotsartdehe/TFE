# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:14:21 2020

@author: Gauthier_Rotsart
"""

import numpy as np
import matplotlib.pyplot as plt

#permet de gerer la rotation de la voiture
def rotation(X,Y,theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for i in range(len(X)):
        X[i], Y[i] = R@np.array([[X[i]],[Y[i]]])
    return X,Y

def voiture(d,theta,a,b,N):
    #position du centre de masse
    dx = d*np.cos(theta)
    dy = d*np.sin(theta)
    
    L1 = a - b/4#longueur de la partie rectiligne horizontale
    L2 = b-1#ongueur de la partie rectiligne verticales
    pente = 0.5/(b/8)#inclinaison des courbures du véhicule
    
    vect = np.ones(N)
    
    #géomètrie
    xx1 = np.linspace(dx-L1/2, dx+L1/2,N)
    yy1 = (b/2 + dy)*vect 
    xx2 = np.linspace(dx+L1/2, dx+a/2,N)
    yy2 = -pente*xx2 + b/2 + dy + pente*(dx+L1/2) 
    xx3 = (a/2+dx)*vect 
    yy3 = np.linspace(dy-L2/2, dy+L2/2, N) 
    xx4 = xx1
    yy4= (dy-b/2)*vect
    xx5 = xx2
    yy5 = pente*xx5 + dy-b/2 - pente*(dx+L1/2) 
    xx6 = np.linspace(dx-a/2, dx-L1/2,N)
    yy6 = -pente*xx6 +dy-b/2 +pente*(dx-L1/2) 
    xx7 = (dx-a/2)*vect 
    yy7 = yy3 
    xx8 = xx6
    yy8 = pente*xx8 + dy+b/2 -pente*(dx-L1/2) 
    plt.figure()
    plt.scatter(0,0,c = 'red')
    
    X = np.zeros(8*N)
    Y = np.zeros(8*N)
    
    X[0:N] = xx1
    Y[0:N] = yy1
    X[N:2*N] = xx2
    Y[N:2*N]= yy2
    X[2*N:3*N] = xx3
    Y[2*N:3*N] = yy3
    X[3*N:4*N] = xx4
    Y[3*N:4*N] = yy4
    X[4*N:5*N] = xx5
    Y[4*N:5*N] = yy5
    X[5*N:6*N] = xx6
    Y[5*N:6*N] = yy6
    X[6*N:7*N] = xx7
    Y[6*N:7*N] = yy7
    X[7*N:8*N] = xx8
    Y[7*N:8*N] = yy8
    
    #si jamais on veut un peu de bruit
    #X += 0.1*np.random.randn(8*N).T
    #Y += 0.1*np.random.randn(8*N).T
    
    #METTRE ICI LA ROTATION SI IL FAUT
    #X,Y = rotation(X,Y,np.pi/2)
    
    plt.scatter(X,Y,c = 'black')
    
    #calcul du point le plus proche
    distance = np.zeros(8*N)
    for i in range(len(distance)):
        distance[i] = np.sqrt((X[i]**2) + Y[i]**2)
    indice2 = np.argmin(distance)
    x1 = X[indice2]
    y1 = Y[indice2]
    
    #calcul du point spéculaire
    pente_N = np.zeros(8*N-1)
    direction = np.zeros(8*N-1)
    for i in range(len(direction)):#calcul des pentes
        if(X[i] == X[i+1]):#droite verticale 
            pente_N[i] = np.inf
        elif(Y[i] == Y[i+1]):#droite horizontale
            pente_N[i] = 0
        else:
            pente = (Y[i+1]-Y[i])/(X[i+1]-X[i])
            pente_N[i] = -1/pente
        direction[i] = np.tan(theta) - pente_N[i]

    indice = np.argmin(direction)
    xx = []
    yy = []
    for i in range(len(direction)):#ensemble de points qui ont des normales paralleles
        if ((direction[i] == direction[indice]) and X[i] < dx):
            xx.append(X[i])
            yy.append(Y[i])
    pente2 = np.zeros(len(xx))
    for i in range(len(xx)):
        pente2[i] = np.abs(np.tan(theta)*xx[i]-yy[i])#A CHANGER SI LA CAMERA N EST PAS EN (0,0)
    index = np.argmin(pente2)
    xxx = xx[index]
    yyy = yy[index]
    
    x = X[indice]
    y = Y[indice]
    
    
    #plt.scatter(xx,yy,c = 'yellow')
    plt.scatter(x1,y1,c='green')
    plt.scatter(xxx,yyy,c='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    return xxx,yyy
