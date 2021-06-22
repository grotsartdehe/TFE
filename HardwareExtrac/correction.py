
"""
Created on Fri Apr  9 16:00:02 2021

@author: kdesousa
"""
import pandas as pd
import numpy as np

def correctionAngle(vect2,dx = 0.022,dy = 0.04,f0=24e9):
    
    theta_rad,phi_rad = vect2[5:7]
    theta_cam,phi_cam = vect2[1:3]
    
    
    
    lam = 3e8/f0 
    ambphi = lam*180/(dx*np.pi) #32.554420177887685 
    ambtheta = lam*180/(dy*np.pi)#17.904931097838226
    nx = np.rint((phi_cam - phi_rad)/ambphi)
    nz = np.rint((theta_cam - theta_rad)/ambtheta)
    
    phi_rad += (nx)*ambphi
    theta_rad += (nz)*ambtheta
    
    vectcor = vect2
    vectcor[5:7] = [theta_rad,phi_rad]
    return vectcor    

def correctionvitesse(vect1,vect2,dt = 1/30):
    
    vectcor = vect2
    d = vect2[3]
    #d1=vect1[count,4]
    v = vect2[6]
    theta1 = vect1[4]*np.pi/180
    theta2 = vect2[4]*np.pi/180
    
    phi1 = vect1[5]*np.pi/180
    phi2 = vect2[5]*np.pi/180
    #print(vect2[m,0])
    vabs = np.sqrt(v**2 + (d*(theta2 - theta1)/dt)**2 + (d*np.sin(theta2)* (phi2 - phi1)/dt)**2)
    
    vectcor[6] = vabs
    
    return vectcor
# 

def correction(vect1,vect2,dt = 1/30):
    vect1 = correctionAngle(vect1)
    vect2 = correctionAngle(vect2)
    if vect1 is None:
        return vect2
    if vect2 is None:
        return vect2
    return correctionvitesse(vect1,vect2,dt = dt)

if __name__ == '__main__':
   dataxee = pd.read_csv('data_est.csv',sep=';').values
   data = pd.read_csv('data.csv')
   test= [ 30.99052613, 92.62386709 ,-8.76805152,  5.72071128,32.1  ,     74.44865824, 25.37918252,  6.02154732]
   test1 = [ 69.51930432, 84.33944902, -8.49572002,  4.87020572,73.2  ,      84.27209838, 25.01496208,  5.04508018]
   test2 = [56.9007908 , 85.8920155,  -3.20535181, -2.78865823,60.    ,     85.84774829, 30.80430528 ,-3.09214592 ]
   # z = correction(test2)
   
   
   
   