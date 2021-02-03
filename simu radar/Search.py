#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:54:36 2021

@author: kdesousa
"""
import numpy as np

def Searchdv(Z,elem,classcar=0):
    result = []
    Z = np.array(Z)
    row = np.array(Z).shape[0]
    col = np.array(Z).shape[1]
    for i in range(elem):
        x = np.argmax(Z)
        result.append(x)
        
        #Z[x%row,x//row]=0
        Z = getzeroed(Z,x%row,x//row,classcar)
    return result

"""
X = np.array([[1,5,8,4],[4,8,3,9],[4,5,6,5]])
print(X.argsort())"""
def getzeroed(Z,i,j,classcar=0):
    for k in range(1):
        Z[i+k-5,j-10:j+10]=0
      
    return Z
if __name__ == "__main__":
    Z = np.array([[4,5,6,80,9],[78,28,92,5,6]])
    print(Z.shape)
    mol = Searchdv(Z, 2)       
    print(mol)
    
def Searchangle(Z,elem):
    result = []
    Z = np.array(Z)
    row = np.array(Z).shape[0]
    col = np.array(Z).shape[1]
    for i in range(elem):
        x = np.argmax(Z)
        result.append(x)
        
        #Z[x%row,x//row]=0
        Z = getzeroed(Z,x%row,x//row)
    return result
    
    