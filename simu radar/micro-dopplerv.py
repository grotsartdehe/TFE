
"""
Created on Mon Jun  7 09:23:58 2021

@author: kdesousa
"""
import numpy as np
import matplotlib.pyplot as plt
N = 512
theta = np.linspace(0,np.pi,N)
phi = np.linspace(-np.pi,np.pi,N)
gamma = np.linspace(-np.pi,np.pi,N)
theta,phi = np.meshgrid(theta,phi)
y = np.tan(theta)*np.cos(phi)
plt.figure()
plt.countourf(theta,phi,y)