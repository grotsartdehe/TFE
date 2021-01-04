# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 12:50:18 2020

@author: Kdesousa
"""
import numpy as np
import matplotlib.pyplot as plt
x = 0
y = 0
dux = 2
duy= 1
#data = np.linspace()
N = 1000
#thetas = np.outer(np.linspace(-np.pi/2,np.pi/2,500), np.ones(500))
thetas = np.outer(np.random.uniform(0,2*np.pi,N), np.ones(N))
m = np.outer(np.linspace(-1,1,N), np.ones(N))
n = m.copy().T
R = np.zeros(shape = m.shape)
ones = (m/dux)**2 + (n/duy)**2 <= 1


x, y = np.mgrid[-1:1:0.01, -1:1:.01]
pos = np.dstack((x, y))
rv1 = np.random.multivariate_normal(mean=[0, 0], cov=[[0.1, 0],[0, 0.1]])
#plt.contourf(x, y, rv1.pdf(pos), cmap='magma')
"""
size = 100
sigma_x = 5
sigma_y = 2.

x = np.linspace(-10, 10, size)
y = np.linspace(-10, 10, size)

x, y = np.meshgrid(x, y)
#z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
     #+ y**2/(2*sigma_y**2))))
z = np.random.multivariate([0,0],[[sigma_x ,0],[0,sigma_y]],(size,size))
plt.contourf(x, y, z, cmap='Blues')
plt.colorbar()
plt.show()

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X and Y
N = 40
X = np.linspace(-2, 2, N)
Y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(X, Y)
dux = 0.36
duy=0.56
# Mean vector and covariance matrix
mu = np.array([0., 0.])
Sigma = np.array([[ dux , 0], [0,  duy]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

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

# The distribution on the variables X, Y packed into pos.

plt.contourf(X,Y,Z+Z1)
plt.colorbar()
