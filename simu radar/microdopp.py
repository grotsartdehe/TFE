#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:31:17 2021

@author: kdesousa
"""
import numpy as np
import matplotlib.pyplot as plt
N = 512
theta = np.linspace(-np.pi,np.pi,N)
phi = np.linspace(-np.pi,np.pi,N)
gamma = np.linspace(-np.pi,np.pi,N)
theta,phi = np.meshgrid(theta,phi)
u = np.arctan(-np.tan(theta)*np.cos(phi))
unorm = np.sqrt(1 + u**2)
vmin = (1-u/unorm)*np.sin(theta)*np.cos(phi) + 1/unorm * np.cos(theta)
vmax  =(1+u/unorm)*np.sin(theta)*np.cos(phi) - 1/unorm * np.cos(theta)
plt.figure()
plt.contourf(theta*180/np.pi,phi*180/np.pi,vmin)
plt.colorbar()
