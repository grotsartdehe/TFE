# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:48:18 2021

@author: Kevin De Sousa & Gauthier Rotsart De Hertaing
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as ss
from numpy import pi


def voiture(d,theta,a,b):
    """
Voiture calcule le point speculaire renvoyé par le chassis d'une classe de vehicule'
"""
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
    # plt.figure()
    # plt.scatter(xx1+dx,yy1+dy, marker = '.', c = 'black')
    # plt.scatter(xx2+dx,yy2+dy, marker = '.', c = 'black')
    # plt.scatter(xx3+dx,yy3+dy, marker = '.', c = 'black')
    # plt.scatter(xx4+dx,yy4+dy, marker = '.', c = 'black')
    # plt.scatter(xx5+dx,yy5+dy, marker = '.', c = 'black')
    # plt.scatter(xx6+dx,yy6+dy, marker = '.', c = 'black')
    # plt.scatter(xx7+dx,yy7+dy, marker = '.', c = 'black')
    # plt.scatter(xx8+dx,yy8+dy, marker = '.', c = 'black')
    # plt.scatter(0,0,c = 'red')

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

    #plt.scatter(x,y,c = 'yellow')
    slope = np.zeros(N*8)
    for j in range(0,8):
        slope[j*N:(j+1)*(N)-2] = -1/(( Y[j*N+1:(j+1)*N-1] - Y[j*N:(j+1)*N-2]) /(X[j*N+1:(j+1)*N-1]-X[j*N:(j+1)*N-2]))

    eval0 = slope*X + Y
    indic = np.argmin(np.abs(eval0))
    xmin = X[indic]
    ymin = Y[indic]
   # plt.scatter(xmin,ymin,c = 'green')


    return x,y,slope



N = 100
X = np.linspace(-1, 1, N)
Y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(X, Y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
theta = np.pi/4
phi = np.pi/8



def ambiguite(X,Y,pos,theta,phi,dux=0.36,duy=0.56):
    """
Ambiguité renvoie l'ampltude '
Argument:
            X: abscisse du repère
            Y: ordonnée du repère
            x: abscisse du vrai signal
            y: l'ordonnée du vrai signal
            dux: ecart horizontale entre 2 ambiguités
            duy: écart vertiale entre 2 ambiguités
            renvoie graphe des ambiguités
"""
    
    #plt.figure()
    u0 = np.cos(theta)
    v0 = np.cos(phi)
    k = 2* np.pi*24e9/3e8
    N = 256
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x,y)
    dx = np.array([1 , 0,0])*0.036
    dy = np.array([1,0,1])*0.025
    #trouvez en utilisant donnée ms1
    #mean = 1.4414414-0.6486486j
    #cov = 2942.9300506820273+596.4310917456352j
    mean  = 0 + 0j
    cov = 1 + 1j
    x = np.random.normal(loc = np.real(mean),scale = np.real(cov),size = 3) +\
        1j*np.random.normal(loc = np.imag(mean),scale = np.imag(cov),size = 3)
    
    signal = x*np.exp(1j*k*(u0*dx+v0*dy))
    Z = signal[0]*np.exp(-1j*k*(X*dx[0]+Y*dy[0]))+\
        signal[1]*np.exp(-1j*k*(X*dx[1]+Y*dy[1]))+\
            signal[2]*np.exp(-1j*k*(X*dx[2]+Y*dy[2]))

    """plt.contourf(X,Y,np.abs(Z))
    plt.colorbar()
    plt.title('simu')
    plt.xlabel('ux')
    plt.ylabel('uy')
    plt.show()"""

    return X,Y,Z



def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N



def radar(d,v,theta,long):
    dmax = 80
    vmax = 70
    res_d = 0.274
    res_v = 0.63
    sigma_d = res_d 
    sigma_v = res_v
    d_new = np.random.normal(d, sigma_d)
    v_new = np.random.normal(v,sigma_v)
    N1=math.floor(dmax/res_d) #291
    N2= math.floor(vmax/res_v) #111
    
    N=np.max([N1, N2])
    #pos est une matrice carré donc je suis obligé de faire cette manip
    #pour garder les bonnes proportions je limite le graphe apres
    X = np.linspace(-N/2*res_v, N/2*res_v, N)
    Y = np.linspace(0, dmax, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    mu = np.array([v_new,d_new])
    omega = v*math.cos(theta)#micro-doppler
    cov = np.array([[res_v,0],[0,res_d]])
    Z = multivariate_gaussian(pos,mu , cov)
    pic= np.unravel_index(Z.argmax(), Z.shape)
    for i in [-1,0,1]:
        
        microdoprange= int((omega/sigma_v )+np.random.normal(loc=2+i,scale=3))
        amp = np.array(Z[pic] * np.ones(int(np.abs(microdoprange)/2)))*np.exp(-1)#*np.exp(-np.arange(microdoprange/2)))
        newamp = np.append(amp[::-1],amp[1:])
        Z[pic[0]+i,pic[1]-len(amp): pic[1]+len(amp)-1] += newamp
        
    kernel = np.zeros((10,10))
    dpoints=int(long*np.cos(theta)/res_d)
    kernel[:,5-int(dpoints):5+int(dpoints)]=0.5
    Znew = ss.convolve2d(Z,kernel,mode = 'same')
    """
    plt.xlim((-vmax,vmax))
    plt.contourf(X,Y,Znew)
    plt.colorbar()
    plt.show()
    """
    
    Znew += np.random.normal(size=Znew.size).reshape(Znew.shape[0],Znew.shape[1])/50
    return X,Y,Znew



def RadarGen(classcar,d,v,theta,phi):
    """
RadarGen genere les données simulés du radar
Inputs: classcar: classe du véhicule [int]
        dist: distance radiale du vehicule au radar en m [int] 
        vitesse: vitesse absolu en m/s [int]
        theta: angle d'élevation [radian]
        phi: angle azimultale [radian]

"""
    #Calcul du point spéculaire 
    N = 100
    X = np.linspace(-1, 1, N)
    Y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    listclasses=[[1.7,4],[1.8,5]]
    
    Z1 = np.zeros((291,291))
    Z2= np.complex64( np.zeros((256,256)))
    Xdv = np.zeros((291,291))
    Ydv = np.zeros((291,291))
    """
    (x,y,m)= voiture(d,theta,a,b) 
    d_spec = np.sqrt(x**2 + y**2)
    (Xdv,Ydv,Zdv)= radar(d_spec,v[0],theta[0],b)
    Z1 += Zdv
    (Xthetaphi,Ythetaphi,Zthetaphi)= ambiguite(X,Y,pos,theta[0],phi[0])
    Z2 += Zthetaphi
    """
    
    
    for i in range(len(d)):
        a = listclasses[classcar[i]][0]
        b = listclasses[classcar[i]][1]
        (Xdv,Ydv,rand)= radar(5,v[0],theta[0],b)
        (Xthetaphi,Ythetaphi,zefi)= ambiguite(X,Y,pos,theta[0],phi[0])
        (x,y,m)= voiture(d[i],theta[i],a,b) 
        d_spec = np.sqrt(x**2 + y**2)
        (Xdv,Ydv,Zdv)= radar(d_spec,v[i],theta[i],b)
        Z1 += Zdv
        (Xthetaphi,Ythetaphi,Zthetaphi)= ambiguite(X,Y,pos,theta[i],phi[i])
        Z2 = Zthetaphi
        
    plt.figure()
    plt.contourf(Xdv,Ydv,Z1)
    plt.xlim(-70,70)
    plt.title('Simulation heatmap (d,v)')
    plt.xlabel('vitesse [km/h]')
    plt.ylabel('distane [m]')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.contourf(Xthetaphi,Ythetaphi,Z2)
    plt.xlabel('ux = cos(theta)')
    plt.ylabel('uy = cos(phi)')
    plt.title("Simu heatmap ux,uy")
    plt.colorbar()
    plt.show()
    

    return Z1,Z2

RadarGen([0,1],[40,60],[-30,50],[np.pi/4,np.pi/7],[np.pi/6,np.pi/7])






