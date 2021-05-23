#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:17:31 2021

@author: kdesousa
"""
import numpy as np
import math

#from kalman_graphics import kalman_draw
import os
import time
import datetime
import pickle


import pandas as pd
import matplotlib.pyplot as plt

def transform(df):
    X = df[:,0]
    Y = df[:,1]
    Z = df[:,2]
    d = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arccos(Z/d) 
    phi = np.arctan2(Y,X)
    return np.array([d,theta,phi,df[:,3]]).T
data_esti = 'data_est_radial.csv'
data = 'data_radial.csv'

A = pd.read_csv(data,sep=';',header=None).values
A0 = pd.read_csv(data_esti,sep=';',header=None).values
A0[:,4] = np.sign(A0[:,-1])* A0[:,4]

A = transform(A[:,1::])
a_rad= transform(A0[:,5::])
a_cam =transform( A0[:,1:5])
cond0 = np.abs(a_cam[:,-1]) < 25 
cond1 = a_cam[:,0] <100
cond = cond0 & cond1
cam = A-a_cam
rad = A-a_rad
var_cam = np.var(cam[cond],axis=0)
var_rad = np.var(rad[cond],axis=0)
