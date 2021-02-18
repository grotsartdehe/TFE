#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:04:28 2021

@author: kdesousa
"""

import numpy as np
import math
import operator
from meta_params import *

class track:

    def __init__(self, i):

        self.id = i
        self.X = np.zeros(6)
        self.c = {'car':0}
       # {"car":0, "person":0, "truck":0, "suv":0,"van":0, "bicycle":0, "motorbike":0, "bus":0, "scooter":0, "ucl":0,"trailer":0, "dog":0, "petit":0, "moyen":0, "grand":0, "init":0.0}
        self.age = 0
        #self.bbox = [0,0,0,0,0,0]
        self.P = np.zeros((6,6))
        self.invisible = 0
        self.K = np.zeros((6,6))
        self.S = np.zeros((6,6))
        self.classe = "init"
        self.height = 0.0
        self.min_det = 0
        self.disper = 0#np.linalg.det(S)
        #Alexis prenait les 2 jeu de données séparement on prend tout ensemble
        # osef du delai
        self.multi = False
        #self.rad = False
        #self.cam = False
        self.obstruction = 0

    def create(self,x,c,r):
        self.X = x
        self.age = 1
        self.classe = c
        
        self.c[c] += 1
        self.P = r
        self.min_det += 1
        self.disper = np.linalg.det(self.P)
        
        
        
    def predict(self, x, p):
        self.X = x
        self.age += 1
        self.P = p
        self.invisible += 1
        self.disper = np.linalg.det(self.P)

    def update(self, x, p, c, det):
        self.X = x
        self.P = p
        self.c[c] += 1
        #self.classe = c
        #self.height = det.H_m
        self.classe = max(self.c.items(), key=operator.itemgetter(1))[0]
        if det:
            self.invisible = 0
            self.min_det += 1
        self.disper = np.linalg.det(self.P)

    def update_sensor(self, s):
        
        if(s == 'cam'):
            self.cam = True
        elif(s == 'radar'):
            self.rad = True
        else:
            print(s)

        if(self.cam and self.rad):
            self.multi = True

    def set_KS(self, k, s):
        self.K = k
        self.S = s
        
class detection:

    def __init__ (self,classcar,x_c,y_c,w,h,d,v,p,timestep): 
         
            self.c0 = classcar
            self.x_c = x_c
            self.y_c = y_c
            self.w = w
            self.h = h
            self.d = d
            self.v = v
            self.wi = 1280
            self.hi = 720
            self.t = timestep
            f = dist_focal
            c = abs((self.y_c-360.0))
            #H_m = hauteur moyenne 
            self.H_m = 3
            #alpha et epsilon les reperes d'Alexis ;)
            self.alpha = math.degrees(math.atan((self.x_c-self.wi/2.0)/f)) + q_alpha 
            self.epsilon = math.degrees(math.atan((self.y_c-self.hi/2.0)/f))
            self.p0 = p
    def get_X(self):

        return np.array([self.alpha, self.epsilon, self.d, 0.0, 0.0, self.v])
            

    def get_confidence(self):
        
        return self.p0

    def get_class(self):
        
        return 'car'

    def get_time(self):

        return self.t

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
             [0.0,0.0,0.0,0.0,0.0,0.0],\
             [0.0,0.0,0.0,0.0,0.0,0.0],\
             [0.0,0.0,0.0,0.0,0.0,1.0]])


        self.Q_back = np.array(
             [[0.01,0.0,0.0,0.0,0.0,0.0],\
             [0.0,0.01,0.0,0.0,0.0,0.0],\
             [0.0,0.0,0.01,0.0,0.0,0.0],\
             [0.0,0.0,0.0,0.002,0.0,0.0],\
             [0.0,0.0,0.0,0.0,0.001,0.0],\
             [0.0,0.0,0.0,0.0,0.0,0.002]])

        self.R_back = np.array(
             [[2.25,0.0,0.0,0.0,0.0,0.0],\
             [0.0,0.3,0.0,0.0,0.0,0.0],\
             [0.0,0.0,25.0,0.0,0.0,0.0],\
             [0.0,0.0,0.0,0.0001,0.0,0.0],\
             [0.0,0.0,0.0,0.0,0.0001,0.0],\
             [0.0,0.0,0.0,0.0,0.0,0.0001]])

        self.R_c_back = np.array(
             [[2.25,0.0,0.0,0.0,0.0,0.0],\
             [0.0,0.3,0.0,0.0,0.0,0.0],\
             [0.0,0.0,6500.0,0.0,0.0,0.0],\
             [0.0,0.0,0.0,10.0,0.0,0.0],\
             [0.0,0.0,0.0,0.0,10.0,0.0],\
             [0.0,0.0,0.0,0.0,0.0,10.0]])

        self.R_r_back = np.array(
             [[270.0,0.0,0.0,0.0,0.0,0.0],\
             [0.0,100.0,0.0,0.0,0.0,0.0],\
             [0.0,0.0,10.3,0.0,0.0,0.0],\
             [0.0,0.0,0.0,10.0,0.0,0.0],\
             [0.0,0.0,0.0,0.0,10.0,0.0],\
             [0.0,0.0,0.0,0.0,0.0,0.8378]])

        self.R0 = init_P


        self.Q = vari_Q

        self.R_c = vari_R_c

        self.R_r = vari_R_r


    def get_matrices(self,):


        return (self.F, self.H, self.Q, self.R_r)


    def get_init_matrice(self):
        
        return self.R0

    def get_params(self):

        return (self.dt, self.gate, self.lamb, self.max_invisible)

class frame:

    def __init__(self):
        self.detections =  []
        self.t = 0.0
        self.pos = False
        self.empty = 0
        self.image = 0

    def add_detection(self, det):
        
        self.t = det.t
        self.detections.append(det)

    def get_detections(self):

        return self.detections
    def get_sensor(self):
        return self.sensor

    def set_time(self,  t):
        
            self.t = t 
       