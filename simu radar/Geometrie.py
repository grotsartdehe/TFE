# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:14:21 2020

@author: Gauthier_Rotsart
"""

import numpy as np
import matplotlib.pyplot as plt

def voiture(d,theta,a,b):
    dx = d*np.cos(theta)
    dy = d*np.sin(theta)
    N = 1000
    L1 = a - b/4#longueur de la partie rectiligne horizontale
    L2 = b-1#ongueur de la partie rectiligne verticales
    pente = 0.5/(b/8)#inclinaison des courbures du véhicule

    vect = np.ones(N)

    #géomètrie
    xx1 = np.linspace(-L1/2, L1/2,N)
    yy1 = b/2*vect #+#0.1*np.random.randn(N).T
    xx2 = np.linspace(L1/2, a/2,N)
    yy2 = -pente*xx2 + b/2 + pente*L1/2 #+ 0.1*np.random.randn(N).T
    xx3 = a/2*vect + 0.1*np.random.randn(N).T
    yy3 = np.linspace(-L2/2, L2/2, N) #+ 0.1*np.random.randn(N).T
    xx4 = xx1
    yy4= -yy1
    xx5 = xx2
    yy5 = pente*xx5 -b/2 - pente*L1/2 #+ 0.1*np.random.randn(N).T
    xx6 = np.linspace(-a/2, -L1/2,N)
    yy6 = -pente*xx6 -b/2 -pente*L1/2 #+ 0.1*np.random.randn(N).T
    xx7 = -a/2*vect#+0.1*np.random.randn(N).T
    yy7 = yy3 #+ 0.1*np.random.randn(N).T
    xx8 = xx6
    yy8 = pente*xx8 + b/2 + L1/2 # + 0.1*np.random.randn(N).T
    plt.figure()
    plt.scatter(xx1+dx,yy1+dy, marker = '.', c = 'black')
    plt.scatter(xx2+dx,yy2+dy, marker = '.', c = 'black')
    plt.scatter(xx3+dx,yy3+dy, marker = '.', c = 'black')
    plt.scatter(xx4+dx,yy4+dy, marker = '.', c = 'black')
    plt.scatter(xx5+dx,yy5+dy, marker = '.', c = 'black')
    plt.scatter(xx6+dx,yy6+dy, marker = '.', c = 'black')
    plt.scatter(xx7+dx,yy7+dy, marker = '.', c = 'black')
    plt.scatter(xx8+dx,yy8+dy, marker = '.', c = 'black')
    plt.scatter(0,0,c = 'red')

    X = np.zeros(8*N)
    Y = np.zeros(8*N)

    X[0:N] = xx1+dx
    Y[0:N] = yy1+dy
    X[N:2*N] = xx2+dx
    Y[N:2*N]= yy2+dy
    X[2*N:3*N] = xx3+dx
    Y[2*N:3*N] = yy3+dy
    X[3*N:4*N] = xx4+dx
    Y[3*N:4*N] = yy4+dy
    X[4*N:5*N] = xx5+dx
    Y[4*N:5*N] = yy5+dy
    X[5*N:6*N] = xx6+dx
    Y[5*N:6*N] = yy6+dy
    X[6*N:7*N] = xx7+dx
    Y[6*N:7*N] = yy7+dy
    X[7*N:8*N] = xx8+dx
    Y[7*N:8*N] = yy8+dy

    distance = np.zeros(8*N)
    for i in range(len(distance)):
        distance[i] = np.sqrt((X[i]**2) + Y[i]**2)
    indice = np.argmin(distance)
    x = X[indice]
    y = Y[indice]

    plt.scatter(x,y,c = 'yellow')
    slope = np.zeros(N*8)
    for j in range(0,8):
        slope[j*N:(j+1)*(N)-2] = -1/(( Y[j*N+1:(j+1)*N-1] - Y[j*N:(j+1)*N-2]) /(X[j*N+1:(j+1)*N-1]-X[j*N:(j+1)*N-2]))

    eval0 = slope*X + Y
    indic = np.argmin(np.abs(eval0))
    xmin = X[indic]
    ymin = Y[indic]
    plt.scatter(xmin,ymin,c = 'green')


    return x,y,slope