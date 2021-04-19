#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:00:02 2021

@author: kdesousa
"""
import pandas as pd
import numpy as np
def correctionAngle(vect2,dx = 0.022,dy = 0.04,f0=24e9):
    
    theta_rad,phi_rad = vect2[5:7]
    theta_cam,phi_cam = vect2[1:3]
    
    pitch= -12.999994000000001
    yaw = -2.5
    
    lam = 3e8/f0 
    ambphi = lam*180/(dx*np.pi) #32.554420177887685 
    ambtheta = lam*180/(dy*np.pi)#17.904931097838226
    nx = np.rint((phi_cam - phi_rad)/ambphi)
    nz = np.rint((theta_cam - theta_rad)/ambtheta)
   
    phi_rad += (nx)*ambphi
    theta_rad += (nz)*ambtheta
    
    # if np.abs(phi_cam - phi_rad) < ambphi:
    #     phi1 = phi_rad
    #     phi1 += ambphi*np.sign(nx)
    #     if np.abs(phi1 - phi_cam) < np.abs(phi_rad - phi_cam):
    #         phi_rad = phi1
            
    # if np.abs(theta_cam - theta_rad) <ambtheta:
    #     theta1 = theta_rad
    #     theta1 += ambtheta * np.sign(nz)
    #     if np.abs(theta1 - theta_cam)> np.abs(theta_rad - theta_cam):
    #         theta_rad = theta1
    #while( phi_cam - phi_rad > ambphi/2):
    #     phi_rad += ambphi
    # while(theta_cam - theta_rad >ambtheta/2) :
    #     theta_rad += ambtheta
        
    #d,theta,phi,v
   
    vectcor = vect2
    vectcor[5:7] = [theta_rad,phi_rad]
    return vectcor

# def correctionvit(vect1,vect2,dt = 1/30):
    
#     """Etape d'assoc"""
#     vectcor = np.zeros((vect2.shape))
#     min_dist = np.ones((vect2.shape[0]))*500
#     index = np.zeros((vect2.shape[0]))
#     for i in range(vect1.shape[0]):
#         for j in range(vect2.shape[0]):
#             l = (vect1[i,0] - vect2[j,0])**2 

#             if min_dist[j] >l:
#                 min_dist[j] = l 
#                 index[j] = j
#     index = np.int_(index)
#     count = 0
#     """"facon 1"""
#     for m  in index:
#         vectcor[count,:] = vect2[m]
#         d = vect2[m,0]
#         d1=vect1[count,4]
#         v = vect2[m,7]
#         theta1 = vect1[count,1]*np.pi/180
#         theta2 = vect2[m,1]*np.pi/180
        
#         phi1 = vect1[count,2]*np.pi/180
#         phi2 = vect2[m,2]*np.pi/180
#         #print(vect2[m,0])
#         vabs = np.sqrt(v**2 + (d*(theta2 - theta1)/dt)**2 + (d*np.sin(theta2)* (phi2 - phi1)/dt)**2)
#         #vabs = v**2 + (d*(theta2 - theta1)/dt)**2 + (d*np.sin(theta2)* (phi2 - phi1)/dt)**2
#         print('vitesse radiale',v)
#         print('vabs',vabs)
#         print('vreal',vect2[m,3])
        # print((phi2 - phi1)/dt)
        # print((theta2 - theta1)/dt)
        #if vabs >15:
            #vectcor[7] = vect1[7]
        #print('vtrue',(vect2[m,3]))
        #print('vestim',vabs)
        #print(vectcor)
        #vectcor[count,7] = vabs
        #count +=1
    # print('methode1',vabs)
    # print('deltaphi',(phi2 - phi1))
    # print( 'dist',d * np.sin(theta2)* (phi2 - phi1)/dt)
    # print('vrad',v)
    # print('vabs',vabs)
    # print('vreal',vect2[3])
    # print(vect2[3])
    
    # """facon2"""
    # vectcor = vect2
    # d = vect2[4]
    
    # vabs = vect2[3]
    # theta1 = vect1[1]*np.pi/180
    # theta2 = vect2[1]*np.pi/180
    # phi1 = vect1[2]*np.pi/180
    # phi2 = vect2[2]*np.pi/180
    # vrad = (vabs**2 - (d*(theta2 - theta1)/dt)**2 - (d * np.sin(theta2)* (phi2 - phi1)/dt)**2)
    # #print('vrad methode2',vrad)
    # vectcor[7] = vrad
    # """facon3"""
    # vectcor = vect2
    # d = vect2[0]
    # v = vect2[3]
    # theta1 = vect1[1]*np.pi/180
    # theta2 = vect2[1]*np.pi/180
    # phi1 = vect1[2]*np.pi/180
    # phi2 = vect2[2]*np.pi/180
    # vabs = np.sqrt(v**2 + (d*(theta2 - theta1)/dt)**2 + (d * np.sin(theta2)* (phi2 - phi1)/dt)**2)
    # vectcor[7] = vabs
    # print('methode3',vectcor[7])
    # #print(d * np.sin(theta2)* (phi2 - phi1)/dt)
    # #print(vectcor)
    # """facon4"""
    # vectcor = vect2
    # d = vect2[4]
    # theta1 = vect2[1]*np.pi/180
    # theta2 = vect2[5]*np.pi/180
    # phi1 = vect2[2]*np.pi/180
    # phi2 = vect2[6]*np.pi/180
    # vabs = np.sqrt(v**2 + (d*(theta2 - theta1)/dt)**2 + (d * np.sin(theta2)* (phi2 - phi1)/dt)**2)
    # vectcor[7] = vabs
    # print('methode4',vabs)
    """facon 5"""
    # print('v_true=',vect2[3])
    # print('v_rad=',vect2[7])
    # print(vect2[0:4],vect2[7])
    # for m  in index:
    #     d = vect2[[m,0]
        
    #     theta2 = vect2[m,1]*np.pi/180
        
    #     phi2 = vect2[m,2]*np.pi/180
    #     vrad = vect2[m,7]
        
    #     x= d*(np.sin(theta2)*np.cos(phi2))
    #     y=d*(np.sin(theta2)*np.sin(phi2))
    #     z = d*(np.cos(theta2))
    #     rz= -phi2
    #     ry = (np.pi/2 - theta2)
        
    #     Rz =np.array( [[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    #     Ry = np.array([[np.cos(ry),0, np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    #     R = Rz @Ry
    #     #[x,y,z] = R @  np.array([x,y,z])
    #     #print(x,d)
    #     vx = vrad/(np.sin(theta2)*np.cos(phi2))
    #     vy = vrad/(np.sin(theta2)*np.sin(phi2))
    #     vz = vrad/(np.cos(theta2))
    #     rho = np.sqrt(vy**2+vx**2+vz**2 )
    #     res = (vx*x + vy*y + vz*z)/(rho*d)
        
    #     count +=1
    # testing = np.arctan2(vrad,res)
    # vres = vrad*np.cos(testing)
    # #print('PLEASE!!',vres)
    # #[vx,vy,vz] = R @  np.array([vx,vy,vz])
    # angle = np.linspace(0,360,1024)*np.pi/180
    # Rx = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]],dtype=object)
    # #[vx,vy,vz]= Rx @  np.array([vx,vy,vz])
    # #print(vx,vy,vz)
    # res = (vx*x + vy*y + vz*z)
    
    # mini = np.argmin(np.abs(vz))
    # mini2 = np.argmin(np.abs(vy))
    
    # rho = np.sqrt(vy**2+vx**2+vz**2 )
    # res = (vx*x + vy*y + vz*z)/(rho*d)
    # testing = np.arccos(res)
    # #vres = vrad/np.cos(testing)
    # #print('no idea',res)
    
    # theta = np.arccos(vz/rho) 
    # #theta = np.arccos(vz[mini]/rho[mini])
    # #t2 = np.arccos(vz[mini2]/rho[mini2]) 
    # #print('theta=',theta*180/np.pi)
    # #phi = np.arctan2(vy[mini],vx)
    # phi = np.arctan2(vy,vx)
    # #p2 =np.arctan2(vy[mini2],vx)
    # #print('phi',phi*180/np.pi)
    # vabs = (np.sqrt(vrad**2))/(np.cos(phi)*np.sin(theta))
    # #print('vestimé',vabs)
    # #vabs = (np.sqrt(vrad**2))/(np.cos(p2)*np.sin(t2))
    # #print('vestimé',vabs)
    # # vabs =vrad/(np.cos(phi))
    # # print('vestimé',vabs)
    
    # #print('vabs',vabs)
    
    
    #vectcor=vect2
    #vectcor[7] = vabs
    # print(vectcor[7])
    
    
    #return vectcor
def correctionvitesse(vect1,vect2,dt = 1/30):
    
    """Etape d'assoc"""
    vectcor = np.zeros((vect2.shape))
    min_dist = np.ones((vect2.shape[0]))*500
    index = np.zeros((vect2.shape[0]))
    #print(vect1)
    if vect1 is None:
        return vect2
    for i in range(vect1.shape[0]):
        for j in range(vect2.shape[0]):
            l = (vect1[i,0] - vect2[j,0])**2 
    
            if min_dist[j] >l:
                min_dist[j] = l 
                index[j] = j
    index = np.int_(index)
    count = 0
    """"facon 1"""
    for m  in index:
        vectcor[count,:] = vect2[m]
        d = vect2[m,0]
        d1=vect1[count,4]
        v = vect2[m,7]
        theta1 = vect1[count,1]*np.pi/180
        theta2 = vect2[m,1]*np.pi/180
        
        phi1 = vect1[count,2]*np.pi/180
        phi2 = vect2[m,2]*np.pi/180
        #print(vect2[m,0])
        vabs = np.sqrt(v**2 + (d*(theta2 - theta1)/dt)**2 + (d*np.sin(theta2)* (phi2 - phi1)/dt)**2)
        #vabs = v**2 + (d*(theta2 - theta1)/dt)**2 + (d*np.sin(theta2)* (phi2 - phi1)/dt)**2
        #print('vabs',vabs)
        if vabs > 20:
            for n in index:
                if not count == n:
                    
                    d = vect2[m,0]
        
                    v = vect2[m,7]
                    theta1 = vect1[n,1]*np.pi/180
                    theta2 = vect2[m,1]*np.pi/180
        
                    phi1 = vect1[n,2]*np.pi/180
                    phi2 = vect2[m,2]*np.pi/180
                    vabs1 = np.sqrt(v**2 + (d*(theta2 - theta1)/dt)**2 + (d*np.sin(theta2)* (phi2 - phi1)/dt)**2)
                    #print(vabs1)
                    if vabs1 < 20:
                        vabs = vabs1
        #print('vitesse radiale',v)
        # print('vabs',vabs)
        # print('vreal',vect2[m,3])
        # print((phi2 - phi1)/dt)
        # print((theta2 - theta1)/dt)
        #if vabs >15:
            #vectcor[7] = vect1[7]
        #print('vtrue',(vect2[m,3]))
        #print('vestim',vabs)
        #print(vectcor)
        #vectcor[count,7] = vabs
        count +=1

if __name__ == '__main__':
   dataxee = pd.read_csv('data_est.csv',sep=';').values
   data = pd.read_csv('data.csv')
   test= [ 30.99052613, 92.62386709 ,-8.76805152,  5.72071128,32.1  ,     74.44865824, 25.37918252,  6.02154732]
   test1 = [ 69.51930432, 84.33944902, -8.49572002,  4.87020572,73.2  ,      84.27209838, 25.01496208,  5.04508018]
   test2 = [56.9007908 , 85.8920155,  -3.20535181, -2.78865823,60.    ,     85.84774829, 30.80430528 ,-3.09214592 ]
   # z = correction(test2)
   
   