#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:35:42 2021

@author: kdesousa
"""
import numpy as np
class heatmap:
        def __init__(self,i): 
            
            self.counter = i
        def store(self,dv,thetaphi):
            self.Zdv = dv
            self.Zambi = thetaphi
            #self.theta = np.append(self.theta,theta)
            #self.phi = np.append(self.phi,phi)
        def getdata(self):
            return self.Zdv,self.Zambi
        def getcounter(self):
            return self.counter
       
class ambig:
    def __init__(self,Zambi, correspondance):
        #self.counter = 0
        self.Z = Zambi
        self.cores = correspondance
    def getZ(self):
        return self.Z
    def getCores(self):
        return self.cores
    
