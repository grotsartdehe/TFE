# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:47:44 2020

@author: Gauthier_Rotsart
"""


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Define all the parameters 

# Matrices A,Bu, G and C 

dt = 0.2 # seconds

A = np.array([[1., dt, 0., 0.],   
              [0., 1., 0., 0.], 
              [0., 0., 1., dt], 
              [0., 0., 0., 1.]])  

g = 9.81 #meter per second^2

Bu = np.array([[0., 0., 0., -g*dt]]).T

C = np.array([[1., 0., 0., 0.],
              [0., 0., 1., 0.]])

G =  np.array([[1., 0., 0., 0.],   
              [0., 1., 0., 0.], 
              [0., 0., 1., 0.], 
              [0., 0., 0., 1.]])  


# Matrices Q, R and P_0

sigma_1 = 0.5
sigma_2 = 0.5
sigma_3 = 1.5
sigma_4 = 0.5

Q = np.array([[sigma_1**2, 0, 0., 0.],   
              [0., sigma_2**2, 0., 0.], 
              [0., 0., sigma_1**2, 0], 
              [0., 0., 0., sigma_2**2]]) 

R = np.array([[sigma_3**2, 0],   
              [0., sigma_3**2]]) 

P_0 = np.array([[sigma_4**2, 0, 0., 0.],   
              [0., 0, 0., 0.], 
              [0., 0., sigma_4**2, 0], 
              [0., 0., 0., 0]]) 

# Initial states

x0bar = 15    # meter
y0bar = 15    # meter
vx0bar = 70   # meter per second
vy0bar = 70   # meter per second

x0barvec = np.array([[x0bar, vx0bar, y0bar, vy0bar]]).T

# Number of time steps

Nsteps = 75            
Xdim = A.shape[0]#dimension de Xk
Ydim = R.shape[0]#dimension de Yk

# Function to generate the trajectory and the measurements

def traj_and_mes(A,Bu,C,G,Q,R,x0vec,Nsteps):
    
    # Outputs :
    # - States, a Nsteps x 4 array containing the states at all time indexes 
    # - Measurements, a Nsteps x 2 array containing the observations at all time indexes 
    
    States = np.zeros((Nsteps,Xdim)) 
    Measurements = np.zeros((Nsteps,Ydim))
    #dt = A[0,1]
    for i in range(Nsteps):
        if i == 0:
            States[0] = x0vec.T  # [x vx y vy].T
        else:
            wk = np.random.multivariate_normal(np.zeros(Xdim).T,Q)            
            vk = np.random.multivariate_normal(np.zeros(Ydim).T,R)
            States[i,:]= A @ States[i-1,:] + Bu.T + G @ wk
            Measurements[i,:] = C @ States[i,:] + vk
    
    return States, Measurements  
States, Measurements  = traj_and_mes(A,Bu,C,G,Q,R,x0barvec,Nsteps)


def Kalman_predictor(A,Bu,C,G,Q,R,P_0,x0barvec,Measurements,Nsteps):
    
    # Output : 
    # - Predicted_states, a Nsteps x 4 array containing the states at all time indexes
    Predicted_states = np.zeros((Nsteps,Xdim))
    
    x0 = np.random.multivariate_normal(np.squeeze(x0barvec),P_0)
    Predicted_states[0,:]= x0   
    for i in range(Nsteps-1):
        if i == 0:
            den = np.linalg.inv(C @ P_0 @ C.T + R)
            K = A @ P_0 @ C.T @ den
            P1 = A @ P_0 @ A.T + G @ Q @ G.T - K @ C @ P_0 @ A.T
            Predicted_states[1,:]= ( A - K @ C) @ x0 + Bu.T + K @ Measurements[0,:]
            
        else:
            den = np.linalg.inv(C @ P1 @ C.T + R)
            K = A @ P1 @ C.T @ den
            P1 = A @ P1 @ A.T + G @ Q @ G.T - K @ C @ P1 @ A.T
            Predicted_states[i+1,:]= ( A - K @ C) @ Predicted_states[i,:] + Bu.T + K @ Measurements[i,:]
 
 
    return Predicted_states
Pred = Kalman_predictor(A,Bu,C,G,Q,R,P_0,x0barvec,Measurements,Nsteps)


def Kalman_filter(A,Bu,C,G,Q,R,P_0,x0barvec,Measurements,Nsteps,Predicted_states):
    
    # Output : 
    # - Filtered_states, a Nsteps x 4 array containing the estimated states at all time indexes
    Filtered_states = np.zeros((Nsteps,Xdim))
    for i in range(Nsteps):
        if i ==0:
            den = np.linalg.inv(C @ P_0 @ C.T + R)
            Kf = P_0 @ C.T @ den
            K = A @ P_0 @ C.T @ den
            P1 = P1 = A @ P_0 @ A.T + G @ Q @ G.T - K @ C @ P_0 @ A.T
            #rand = np.random.multivariate_normal([0,0],R)
             
            Filtered_states[0,:]= Predicted_states[0,:] + Kf @ (Measurements[0,:] - C @ Predicted_states[0,:])# - rand)
        else:
            den = np.linalg.inv(C @ P1 @ C.T + R)
            Kf = P1 @ C.T @ den
            K = A @ P1 @ C.T @ den
            P1 =  A @ P1 @ A.T + G @ Q @ G.T - K @ C @ P1 @ A.T
            #rand = np.random.multivariate_normal([0,0],R)
            Filtered_states[i,:]= Predicted_states[i,:] + Kf @ (Measurements[i,:] - C @ Predicted_states[i,:])#- rand)
             
    return Filtered_states
#Filt = Kalman_filter(A,Bu,C,G,Q,R,P_0,x0barvec,Measurements,Nsteps,Predicted_states)
Filt = Kalman_filter(A,Bu,C,G,Q,R,P_0,x0barvec,Measurements,Nsteps,Pred)

plt.figure(figsize=(20, 10))
#N = int((Nsteps-1)/2)

plt.plot(States[:,0],States[:,2],'g')
plt.plot(Measurements[:,0],Measurements[:,1],'ro')
plt.plot(Pred[:,0],Pred[:,2],'b+')
plt.plot(Filt[:,0],Filt[:,2],'rx')
plt.legend(('States','Measures', 'Predictor',"Filter"))
