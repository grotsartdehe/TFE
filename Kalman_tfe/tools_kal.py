#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:19:58 2021

@author: kdesousa
"""
import numpy as np
from meta_para import *
class kalman_params:

    def __init__(self):

        self.dt = delta_time 
        self.gate = min_dist_gate 
        self.lamb = lambda_fp
        self.max_invisible = max_invisible_no_obs #int(4*(1/self.dt))
        
        self.F = np.array(
             [[1.0,0.0,0.0,self.dt,0.0,0.0],\
             [0.0,1.0,0.0,0.0,self.dt,0.0],\
             [0.0,0.0,1.0,0.0,0.0,self.dt],\
             [0.0,0.0,0.0,1.0,0.0,0.0],\
             [0.0,0.0,0.0,0.0,1.0,0.0],\
             [0.0,0.0,0.0,0.0,0.0,1.0]])

        self.H = np.array(
             [[1.0,0.0,0.0,0.0,0.0,0.0],\
             [0.0,1.0,0.0,0.0,0.0,0.0],\
             [0.0,0.0,1.0,0.0,0.0,0.0],\
             [0.0,0.0,0.0,1.0,0.0,0.0],\
             [0.0,0.0,0.0,0.0,0.0,0.0],\
             [0.0,0.0,0.0,0.0,0.0,0.0]])



        # self.Q_back = np.array(
        #      [[0.01,0.0,0.0,0.0,0.0,0.0],\
        #      [0.0,0.01,0.0,0.0,0.0,0.0],\
        #      [0.0,0.0,0.01,0.0,0.0,0.0],\
        #      [0.0,0.0,0.0,0.002,0.0,0.0],\
        #      [0.0,0.0,0.0,0.0,0.001,0.0],\
        #      [0.0,0.0,0.0,0.0,0.0,0.002]])

        # self.R_back = np.array(
        #      [[25.0,0.0,0.0,0.0,0.0,0.0],\
        #      [0.0,0.3,0.0,0.0,0.0,0.0],\
        #      [0.0,0.0,2.25,0.0,0.0,0.0],\
        #      [0.0,0.0,0.0,0.0001,0.0,0.0],\
        #      [0.0,0.0,0.0,0.0,0.0001,0.0],\
        #      [0.0,0.0,0.0,0.0,0.0,0.0001]])

        # self.R_c_back = np.array(
        #      [[6500.0,0.0,0.0,0.0,0.0,0.0],\
        #      [0.0,0.3,0.0,0.0,0.0,0.0],\
        #      [0.0,0.0,2.25,0.0,0.0,0.0],\
        #      [0.0,0.0,0.0,10.0,0.0,0.0],\
        #      [0.0,0.0,0.0,0.0,10.0,0.0],\
        #      [0.0,0.0,0.0,0.0,0.0,10.0]])

        # self.R_r_back = np.array(
        #      [[100,0.0,0.0,0.0,0.0,0.0],\
        #      [0.0,100.0,0.0,0.0,0.0,0.0],\
        #      [0.0,0.0,5.0,0.0,0.0,0.0],\
        #      [0.0,0.0,0.0,150,0.0,0.0],\
        #      [0.0,0.0,0.0,0.0,10.0,0.0],\
        #      [0.0,0.0,0.0,0.0,0.0,10.0]])

        self.P = init_P


        self.Q = vari_Q

        #self.R = self.R_c_back @ np.linalg.inv(self.R_r_back + self.R_c_back) @ self.R_r_back
        self.R_inv = np.linalg.inv(vari_R_c + vari_R_r)


    def get_matrices(self):

        #print(weights[0,:] @ self.R_c_back + weights[1,:]@self.R_r_back)
        return (self.F, self.H, self.Q, self.R_inv, vari_R_c,vari_R_r)


    def get_init_matrice(self):
        
        return self.P

    def get_params(self):
        
        return (self.dt, self.gate, self.lamb, self.max_invisible)
class track:

    def __init__(self, i):

        self.id = i
        self.X = np.array([6*[0.0]])
        self.c = {"car":0, "person":0, "truck":0, "suv":0,"van":0, "bicycle":0, "motorbike":0, "bus":0, "scooter":0, "ucl":0,"trailer":0, "dog":0, "petit":0, "moyen":0, "grand":0, "init":0.0}
        self.age = 0
        #self.bbox = [0,0,0,0,0,0]
        self.P = np.array(6*[6*[0.0]])
        self.invisible = 0
        self.K = np.array(6*[6*[0.0]])
        self.S = np.array(6*[6*[0.0]])
        self.classe = "init"
        self.height = 0.0
        self.min_det = 0
        self.disper = 0#np.linalg.det(S)
        self.multi = False
        self.rad = False
        self.cam = False
        self.obstruction = 0

    def create(self,x,p):
    
        self.X = x
        self.age = 1
        
        self.P = p
        self.min_det += 1
        self.disper = np.linalg.det(self.P)


    def predict(self, x, p):
        self.X = x
        self.age += 1
        self.P = p
        self.invisible += 1
        self.disper = np.linalg.det(self.P)

    def update(self, x, p, det):
        self.X = x
        self.P = p
        #self.c[c] += 1
        #self.classe = c
        #self.height = det.H_m
        #self.classe = max(self.c.items(), key=operator.itemgetter(1))[0]
        if det:
            self.invisible = 0
            self.min_det += 1
        self.disper = np.linalg.det(self.P)
    def set_KS(self, k, s):
        self.K = k
        self.S = s