# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:42:25 2020

@author: Gauthier_Rotsart
"""


import numpy as np
import matplotlib.pyplot as plt
N = 100
X = np.linspace(-1, 1, N)
Y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(X, Y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
dux = 0.36
duy = 0.56
theta = np.pi/4
phi = np.pi/8


"""
Argument:
            X: abscisse du repère
            Y: ordonnée du repère
            x: abscisse du vrai signal
            y: l'ordonnée du vrai signal
            dux: ecart horizontale entre 2 ambiguités
            duy: écart vertiale entre 2 ambiguités
            renvoie graphe des ambiguités
"""
def ambiguite(X,Y,pos,theta,phi,dux,duy):
    x =np.sin(phi)* np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    ambiguite_x1= np.arange(x,1.5,dux)
    ambiguite_x = np.append(np.flip(np.arange(x,-1.5,-dux)), ambiguite_x1[1:])
    print(ambiguite_x)
    ambiguite_y1 = np.arange(y,1.5,duy)
    ambiguite_y = np.append(np.flip(np.arange(y,-1.5,-duy)), ambiguite_y1[1:])
    Z = np.zeros(shape=(X.shape))
    for i in ambiguite_x:
        for j in ambiguite_y:

            Z1 = multivariate_gaussian(pos, np.array([i,j]), np.array([[dux/20,(dux+duy)/200],[(dux+duy)/40, duy/20]]))
            Z = Z1 + Z
    plt.contourf(X,Y,Z)
    plt.colorbar()




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

ambiguite(X,Y,pos,theta,phi,dux,duy)

