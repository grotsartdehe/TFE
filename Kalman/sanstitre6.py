#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:56:23 2021

@author: kdesousa
"""
import pandas as pd 
import numpy as np
data = pd.read_csv('data.csv',sep = ';',header = None).values
data_esti = pd.read_csv('data_est.csv',sep = ';',header= None).values
d_esti = data_esti[:,(1,5)]
phi_esti = data_esti[:,(2,6)]
theta_esti = data_esti[:,(3,7)]
v_esti= data_esti[:,(4,8)]
d= data[:,1]
theta = data[:,2]
phi = data[:,3]
v = data[:,4]
w1 = np.linalg.solve(d_esti,d.T)