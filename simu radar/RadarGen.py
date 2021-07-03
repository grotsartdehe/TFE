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
import pandas as pd
import warnings
from tools import ambig
from Search import *
from correction import*
warnings.filterwarnings("ignore", category=RuntimeWarning) 

"caractéristiques du radar simulé:"
tailles_vehicules =pd.read_csv('vehicule_dimension.csv',sep = ';')
f_s = 3.413e6;
f_0=24e9;
N_s=256;
#f_r=22.1 ;
c = 3e8;
w_0 = 2*np.pi*f_0;
BW = 545.5e6;
largeur = 3
res_d = 0.274
treshold = 50
d = np.arange(256)* (c/(2*BW))
v = np.arange(-128,128,1)*(c*np.pi*f_s/(2*w_0*N_s*256))
d,v= np.meshgrid(d, v)

def voiture(d,theta,phi,xsi,a,b):
    """
calcule le point speculaire renvoyé par le chassis d'une classe de vehicule modélisé par
un OCTOGONE

""" 
    
    
    
    dx = d*np.sin(theta)*np.cos(phi)
    dy = d*np.sin(theta)*np.sin(phi)
    dz = d*np.cos(theta)
    
    N = 200
    L1 = a - b/2#longueur de la partie rectiligne horizontale
    L2 = b-1#ongueur de la partie rectiligne verticales
    pente = 0.5/(b/4)#inclinaison des courbures du véhicule
    
    vect = np.ones(N)

    #géomètrie
    xx1 = np.linspace(-L1/2, L1/2,N)
    yy1 = b/2*vect #+ 0.1*np.random.randn(N).T
    xx2 = np.linspace(L1/2, a/2,N)
    yy2 = -pente*xx2 + b/2 + pente*L1/2 #+ 0.1*np.random.randn(N).T
    xx3 = a/2*vect #+ 0.1*np.random.randn(N).T
    yy3 = np.linspace(-L2/2, L2/2, N) #+ 0.1*np.random.randn(N).T
    xx4 = xx1
    yy4= -yy1
    xx5 = xx2
    yy5 = pente*xx5 -b/2 - pente*L1/2 #+ 0.1*np.random.randn(N).T
    xx6 = np.linspace(-a/2, -L1/2,N)
    yy6 = -pente*xx6 -b/2 -pente*L1/2 #+ 0.1*np.random.randn(N).T
    xx7 = -a/2*vect #+0.1*np.random.randn(N).T
    yy7 = yy3  #+ 0.1*np.random.randn(N).T
    xx8 = xx6
    yy8 = pente*xx8 + b/2 + pente*L1/2  # + 0.1*np.random.randn(N).T
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
    # plt.show()

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
    X += np.random.normal(scale=0.02,size = 8*N)
    Y += np.random.normal(scale=0.02,size = 8*N)
    R = np.array([[np.cos(xsi),-np.sin(xsi)],[np.sin(xsi),np.cos(xsi)]])
    #print(R.shape)
    """for i in range(len(X)):
        
         [X[i],Y[i]] = R.T @ np.array([[X[i],Y[i]]])"""
    X
    X = X + dx
    Y = Y + dy

    
    # plt.figure()
    # plt.scatter(Y,X, marker = '.', c = 'black')
    # plt.scatter(0,0,c = 'red')
    # plt.xlabel('Axe x [m]')
    # plt.ylabel('Axe y [m]')
    # plt.title('Position du châssis dans l\'espace')

    distance = np.zeros(8*N)
    for i in range(len(distance)):
        distance[i] = np.sqrt((X[i]**2) + Y[i]**2)
    indice = np.argmin(distance)
    x = X[indice]
    y = Y[indice]
    #plt.scatter(y,x,c = 'yellow')
    slope = np.zeros(N*8)
    for j in range(0,8):
        slope[j*N:(j+1)*(N)-2] = -1/(( Y[j*N+1:(j+1)*N-1] - Y[j*N:(j+1)*N-2]) /(X[j*N+1:(j+1)*N-1]-X[j*N:(j+1)*N-2]))

    eval0 = slope*X + Y
    indic = np.argmin(np.abs(eval0))
    xmin = X[indic]
    ymin = Y[indic]
    #plt.scatter(ymin,xmin,c = 'green')


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



def ambiguite(theta,phi,dux=0.3125,duy=0.568,cam_number=0):
    """
Ambiguité renvoie carte thermique des angles 
Argument:
            
            renvoie graphe des ambiguités
"""
    
    #plt.figure()
    # if theta <0:
    #      u0 = np.cos(-pi/2 - theta)*np.sin(phi)
    #      v0 = np.sin(-pi/2 - theta)
    # else:
    # ux = np.cos(pi/2- theta)*np.sin(phi)
    # vz = np.sin(pi/2 - theta)
    if cam_number == 1 or cam_number == 2:
        ux = np.cos(pi/2- theta)*np.cos(phi)
        vz = np.sin(pi/2 - theta) 
    else:
        ux = np.cos(pi/2- theta)*np.sin(phi)
        vz = np.sin(pi/2 - theta)
    # vz = np.cos(theta)
    # ux = np.sin(phi)
    #print('ux=',ux,'v0=',vz)
    #check site de mathoworks attention ici theta est utilisé par rapport a 
    # la convention de wiki pedia avec theta partant de l'axe z sur le planxy
    #print(u0,v0)
    k = 2* np.pi*24e9/3e8
    N = 4*256
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x,y)
    dx = np.array([1 , 0,0])*0.022
    dz = np.array([0,0,-1])*0.04
    #trouvez en utilisant donnée ms1
    #mean = 1.4414414-0.6486486j
    #cov = 2942.9300506820273+596.4310917456352j
    mean  = 0 + 0j
    cov = 0.001 + 0.001j
    x = 20 + 20j #+np.random.normal(loc = np.real(mean),scale = np.real(cov),size = 3) +\
        #1j*np.random.normal(loc = np.imag(mean),scale = np.imag(cov),size = 3)
    
    signal = x*np.exp(1j*k*(ux*dx+vz*dz))
    signal += np.random.normal(3)+ 1j*np.random.normal(3)
    #print(signal)
    Z = signal[0]*np.exp(-1j*k*(X*dx[0]+Y*dz[0]))+\
        signal[1]*np.exp(-1j*k*(X*dx[1]+Y*dz[1]))+\
            signal[2]*np.exp(-1j*k*(X*dx[2]+Y*dz[2]))
    m = Z[128:167,128:199]
    # plt.figure()
    #plt.contourf(X[128:167,128:199],Y[128:167,128:199],np.abs(m))
    # plt.title('abs')
    # plt.figure()
    # plt.contourf(X[128:167,128:199],Y[128:167,128:199],np.real(m))
    # plt.title('real')
    # plt.figure()
    # plt.contourf(X[128:167,128:199],Y[128:167,128:199],np.imag(m))
    # plt.title('imag')
    # plt.colorbar()
    # plt.title('simu')
    # plt.xlabel('ux')
    # plt.ylabel('uy')
    # plt.show()
    return np.abs(Z)



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

def putindex(Z,row,col,index,classcar=0,res_d = 0.274):
    "sert a créer associer la postion d'un vehicule (angles) à chaque véhicule"
    large = int(np.floor(2.5/res_d)/2) #longeur voiture ~ 4m + 1m pour marge
    
    Z[row-large:row+large,col-large:col+large]=index
    return Z

def radar(d,v,phi,index,long,largeur,xsi,vabs,f0=24e9):
    #check demo gauthier
    
    k = 2 * pi*f0/3e8
    dmax = 70
    vmax = 70/3.6
    res_d = 0.274
    res_v = 0.63/3.6
    sigma_d = res_d 
    sigma_v = res_v
    #d_new = np.random.normal(d, sigma_d)
    v= np.random.normal(v,sigma_v)
    # N1=math.floor(dmax/res_d) #291
    # N2= math.floor(vmax/res_v) #111
    a = 0.40
    N=256
    #pos est une matrice carré donc je suis obligé de faire cette manip
    #pour garder les bonnes proportions je limite le graphe apres
    dech = np.arange(256)* (c/(2*BW))
    vech = np.arange(-128,128,1)*(c*np.pi*f_s/(2*w_0*N_s*256))
    X, Y = np.meshgrid(vech,dech)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    if d==0 and v ==0:
        Zdv = np.random.normal(size = (X.shape[0],Y.shape[0]))
        return X,Y,Zdv
    mu = np.array([v,d])
    
    #print(omega)#micro-doppler
    Za  = np.zeros((N,N))
   
    
    
    
    cov = np.array([[res_v,0],[0,res_d]])
    Z = multivariate_gaussian(pos,mu , cov)
    
    pic= np.unravel_index(Z.argmax(), Z.shape)
    row,col = pic
    Z += np.ones((1,1))
    Za = putindex(Za,row,col,index)
        
    
    
    """Generation Micro-Doppler new"""
    u = np.arctan(-np.tan(theta)*np.cos(phi))
    unorm = np.sqrt(1 + u**2)
    vmin = np.sign(v)*vabs*(1-u/unorm)*np.sin(theta)*np.cos(phi-xsi) + 1/unorm * np.cos(theta)
    vmax  =np.sign(v)*vabs*(1+u/unorm)*np.sin(theta)*np.cos(phi-xsi) + 1/unorm * np.cos(theta)
    if vmin < vech[0]:
        vmin = vech[0]
    if vmax > vech[-1]:
        vmax = vech[-1]
    
    indexmin = np.argmin((vech - vmin)**2)
    indexmax = np.argmin((vech-vmax)**2)
    if indexmin > indexmax :
        a = indexmin
        indexmin = indexmax
        indexmax = a
    
    vdoppler = np.zeros(vech.shape)
    vdoppler[indexmin:indexmax]=0.5
    Z[pic[0],:] += vdoppler
    
    
    
    chassis = np.abs(long*np.cos(xsi) + largeur*np.sin(xsi))
   
    dpoints=int(chassis/(res_d*2))
    kernel = np.zeros((dpoints,3))
    
    
    
    m = np.arange(-1,2)*5
    m = - np.abs(m)
    
    kernel[:,2]=1/dpoints
    kernel[0,:]=2*np.exp(m)/dpoints
    kernel[-1,:]=2*np.exp(m)/dpoints
    Znew =ss.convolve2d(Z,kernel,mode = 'same')
    
    
    """plt.xlim((-vmax,vmax))
    plt.contourf(X,Y,Znew)
    plt.colorbar()
    plt.show()"""
    
    
    #Znew += np.random.normal(scale = 0.15,size=Znew.size).reshape(Znew.shape[0],Znew.shape[1])
    
    return X,Y,Znew,Za


def LookDimClass(stringcar):
    "trouve la dim du véhicule"
    count = 0
    for i in tailles_vehicules['ID']:
        if not stringcar.find(i) ==-1:
            return tailles_vehicules['Dim_X'][count]\
                ,tailles_vehicules['Dim_Y'][count]
        count+=1
    return tailles_vehicules['Dim_X'][0]\
                ,tailles_vehicules['Dim_Y'][0]
    
    
    
def RadarGen(classcar,d,v,theta,phi,xsi,vabs):
    """
RadarGen genere les données simulés du radar
Inputs: classcar: classe du véhicule [string]
        dist: distance radiale du vehicule au radar en m [int] 
        vitesse: vitesse absolu en m/s [int]
        theta: angle azimutale [radian]
        phi: angle d'élevation [radian]
        xsi: orientation du véhicule

"""
    
    #Calcul du point spéculaire 
    N = 100
    X = np.linspace(-1, 1, N)
    Y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    
    Zamb = np.zeros((256,256))
    Z1 = np.zeros((256,256))
    #Z2= np.complex64( np.zeros((len(d.index),256,256)))
    Xdv = np.zeros((256,256))
    Ydv = np.zeros((256,256))
    
    
    j = 0
    #appartenance = np.zeros((len(d.index),2))
    
    if len(d)==0:
        
        Xdv,Ydv,Zdv,Za = radar(0,0,0,0,0,0,0,0)
        #print("entrer dans la boucle d.index ==0")
    else:
        for i in range(len(d)):
            
            a,b = LookDimClass(classcar[i])
           
            (x,y,m)= voiture(d[i],theta[i],phi[i],xsi[i],a/100,b/100) 
            d_spec = np.sqrt(x**2 + y**2)
            (Xdv,Ydv,Zdv,Za)= radar(d_spec,v[i],phi[i],i+2,a/100,b/100,xsi[i],vabs[i])
            Z1 += Zdv
            Zamb += Za
            Zamb[Zamb>i+3]=i+2
            #(#Xthetaphi,Ythetaphi,Zthetaphi)= ambiguite(theta[i],phi[i])
            #Z2[j,:,:] = Zthetaphi
            #appartenance[j,0] = d_spec
            #appartenance[j,1] = d_spec
            #j+=1
    #heatambi = ambig(Z2,appartenance)
    "plot DV heatmap"
    # plt.figure()
    # d = np.arange(256)* (c/(2*BW))
    # v = np.arange(-128,128,1)*(c*np.pi*f_s/(2*w_0*N_s*256))
    # #Z1 = (Z1 - np.mean(Z1)/np.std(Z1))
    # #plt.contourf(v,d,20*np.log(Z1))
    # plt.contourf(v,d,Z1)
    # plt.title('Simulation de carte thermique distance-vitesse ')
    # plt.xlabel('vitesse [m/s]')
    # plt.ylabel('distance [m]')
    # plt.colorbar(label = 'Amplitude [dB]')
    # plt.show()
    # #plot ambiguité map
    # plt.figure()
    # plt.colorbar(label = 'Amplitude [dB]')
    # X = np.linspace(-1,1,row)
    # Y  = np.linspace(-1,1,col)
    # plt.contourf(X,Y,20*np.log(Zamb))
    # plt.xlabel('u [ ]')
    # plt.ylabel('v [ ]')
    # plt.title('Carte thermique des angles pour cas réel')
    # plt.colorbar(label = 'Amplitude [dB]')
    # plt.colorbar()
    # plt.show()
    
    return Z1,Zamb# ,heatambi

# Znew,Za = RadarGen(['5458'],[47.5],[10],[93.83*np.pi/180],[68.866*np.pi/180],[np.pi/6],[15])
# d,v,row,col = Searchdv(Znew,256,256)
# print(d,v)
# amb = ambiguite(91,-3,dux=0.3125,duy=0.568,cam_number=0)
# a,b = (Searchangle(amb))
# print(a*180/np.pi,b*180/np.pi)
# plt.figure()

# X = np.linspace(-1,1,1024)
# Y  = np.linspace(-1,1,1024)
# # plt.contourf(X,Y,20*np.log(amb))
# # plt.xlabel('u [ ]')
# # plt.ylabel('v [ ]')
# # plt.title('Simulation  de carte thermique des angles ')
# # plt.colorbar(label = 'Amplitude [dB]')

# plt.show()
# # lam = 3e8/24e9
# # dx=0.022
# # dz=0.04
# # ambx = lam/(dx) #32.554420177887685 
# # ambz = lam/dz #17,55
# # theta = 88.66171256 
# # phi =  -6.49220671
# # print('theta=',theta*np.pi/180,'phi=',phi*np.pi/180)
# # Hope = ambiguite(theta*np.pi/180,phi*np.pi/180)
# # print(Searchangle(Hope))





