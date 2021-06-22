#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:18:39 2021

@author: kdesousa
"""
import numpy as np
import math
from tools_kal import *
#from kalman_graphics import kalman_draw
import os
import time
import datetime
import pickle
from meta_para import *

import pandas as pd
import matplotlib.pyplot as plt
from correction import correctionAngle
import time  
class Timer(object):  
    def start(self):  
        if hasattr(self, 'interval'):  
            del self.interval  
        self.start_time = time.time()  
  
    def stop(self):  
        if hasattr(self, 'start_time'):  
            self.interval = time.time() - self.start_time  
            del self.start_time # Force timer reinit  
def kalman(data_esti, data5):
    """
    

    Parameters
    ----------
    data_esti : string name csv
        csv conteant csv de taille n x 9 
        avec pour chaque colonne : numéro de la frame,distance_camera,theta_cam,phi_cam,v_cam
                                    d_rad,theta_rad,phi_rad,v_rad
    data : string name csv
        csv conteant ground truth a titre de comparaison

    Returns
    -------
    None.

    """
    
    dt, gate, lamb, max_invisible = params.get_params()
    
    truth_gr = pd.read_csv(data5,sep=';',header = None).values
    data = load_data(data_esti)
    data[:,1:5] = transform(data[:,1:5])
    data[:,5::] = transform(data[:,5::])
    truth_gr[:,1::] = transform(truth_gr[:,1::])
    tracks = []
    # plt.figure()
    # plt.title('Trajectoire filtre de Kalman avec paramètres adaptés')
    # plt.xlabel('X(t) [m]')
    # plt.ylabel('Y(t) [m]')
    k1=0
    i = 0
    MSE = np.zeros(int(len(data) ))
    
    maxim =0 
    for index in (range(int(data[0,0]),int(data[-1,0]))):
    #for index in (range(int(1580),int(1680))):
        # if index //50 ==0:
            #print(index)
        tracks = kalman_estimate(tracks, data[data[:,0]==index])
        #if len(tracks) > 0:
        #print(len(tracks))
        maxim = max(maxim,len(tracks))
        count = 0
        for k in tracks:
            #print(k.X)
            x,y,z = getcarte(k.X)
            #plt.scatter(y,x,c = 'blue')
            
        # for m in range(len(truth_gr[truth_gr[:,0]==index])):
        #     inter = truth_gr[truth_gr[:,0]==index]
        #     #print('truth',inter[m,1::])
        #     x1,y1,z1 =getcarte(inter[m,1::])
        #     plt.scatter(x1,y1,c = 'orange')
        #     plt.legend(['trajectoire du fi'trajectoire réelle'])
            
        # inter = truth_gr[truth_gr[:,0]==index]
        # L2,index = assos(tracks,inter[:,1::])
        # p =L2.argsort()
        #print(L2)
        
        #MSE[i] = np.mean(L2[L2<200])
        i += 1
    
    # plt.figure()
    # plt.scatter(range(len(MSE[MSE>0])),MSE[MSE>0])
    # plt.title('MSE per iteration')
    # plt.xlabel('iteration i')
    # plt.ylabel('MSE')
    
    #print("MSE",np.mean(MSE[MSE>0]))
    #Lister.append(np.mean(MSE[MSE>0]))
    return(np.mean(MSE[MSE>0])),maxim
    
    
def assos(tracks,truth):
    mindist = np.ones(len(tracks))*500
    index = np.zeros(len(tracks))
    for i in range(len(tracks)):
        
        x,y,z = getcarte(tracks[i].X)
        for j in range(truth.shape[0]):
            x0,y0,z0 = getcarte(truth[j,1::])
            dist = np.sqrt((x-x0)**2 +(y-y0)**2+(z-z0)**2)
            #dist += np.sqrt((i.X[j,-1] -truth[j,-1] )**2)
            
            if dist < mindist[i]:
                mindist[i]= dist
                #print(i.X)
                #print(truth[j,1::])
                index[i] = j
    
    return mindist,index



def transform(df):
    X = df[:,0]
    Y = df[:,1]
    Z = df[:,2]
    d = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arccos(Z/d) 
    phi = np.arctan2(Y,X)
    return np.array([d,theta,phi,df[:,3]]).T
def load_data(data):
    data = pd.read_csv(data,sep=';',header = None).values
    return data
def kalman_estimate(tracks,detections):
    #print('debut estimate',tracks)
    if len(tracks)==0:
        return housekeep(tracks,detections)
    tracks = predict(tracks)
    if(len(detections) == 0):
        #print('ici?')
        for tr in tracks:
            tr.invisible += 1
        return housekeep(tracks, detections)
    
    tracks, detections, associated, new = associate(tracks, detections)
    #print('tracks',tracks[0].X)
    #print('deterctions',detections)
    #print('association',associated)
    #print('new',new)
    # Update
    tracks, new, detections = update(tracks, detections, associated, new)
    #print('tracks_update',tracks[0].X)
    # Housekeep
    filtered_tracks = housekeep(tracks, detections, new)
    
    #print('tracks_housekeep',tracks[0].X)
    return filtered_tracks
    
    
    
    
    
    
    
    

def housekeep(tracks, detections, new = 0):
    dt, gate, lamb, max_invisible = params.get_params()
    """init"""
    
    if(len(tracks) == 0 and len(detections) != 0):
        new_tracks = []
        for det in range(detections.shape[0]):
            new_t = track(0)
            try:
                F,H,Q,Rinv,R_cam,R_rad = params.get_matrices()
            except:
                F,H,Q,Rinv,R_cam,R_rad = params.get_matrices() #pas compris a quoi ca servait...
            
            a = np.zeros(6)
            a[0:4] = detections[det,1:5]
            b = np.zeros(6)
            b[0:4]= detections[det,5:9]
            
            #X= R_rad @ Rinv @ a + R_cam @ Rinv @ b
            X = Rinv @ (R_rad @ b + R_cam @ a)
            R = Rinv
            
            # print(a)
            # print(b)
            
            P_0 = params.get_init_matrice()
            new_t.create(X,R)
            new_tracks.append(new_t)
            
        tracks.extend(new_tracks)
        
        
        
    """new tracks with unassociated detections"""
    if not new==0 and len(new)>0:
        new_tracks = []
        for n in new:
            det = n[1]
            
            #
            #new_thresh = 0.5#new_thresh_classes[det.get_class()]
            
            if(n[2] > new_thresh):

                new_t = track(0)
                F,H,Q,Rinv,R_cam,R_rad = params.get_matrices()
                
                P_0 = params.get_init_matrice()
                #print('housekeep new tracks with unassos',R_0)
                a = np.zeros(6)
                a[0:4] = detections[det,1:5]
                b = np.zeros(6)
                b[0:4]= detections[det,5:9]
                # R = R_cam @ Rinv @ R_rad
                # X= R_rad @ Rinv @ a + R_cam @ Rinv @ b
                X = Rinv @ (R_rad @ b + R_cam @ a)
                
                R = Rinv
                new_t.create(X, R)
                new_tracks.append(new_t)
                #print(a)
                #print(b)
        # Keep only good tracks
        
        tracks.extend(new_tracks)
    keep_tracks = []
    for tr in tracks:
    
        #thresh = obstruction_thresh

        obs = False

        # for t in tracks:
            
            # thresh = 0#*math.degrees(math.atan(objects_obs_size[t.classe]/(2*t.X[2])))
            # if(abs(tr.X[0] - t.X[0]) < thresh and (tr.X[2] > t.X[2]) and tr.min_det > min_det_obstruction):
            #     if(t.min_det > min_obs_front):
            #         tr.obstruction += 1
            #         obs = True
            #         print('obs')

        if(obs == False):
            tr.obstruction = 0
    
            #else:
            #    tr.obstruction = 0

            #tr.obstruction = max(0, tr.obstruction)

        if(tr.invisible < max_invisible_obs):
            if((tr.invisible < max_invisible_no_obs) or (tr.obstruction > 0)):
                keep_tracks.append(tr)    

    return keep_tracks
    
    return tracks

def predict(tracks):

    F,H,Q,Rinv,R_cam,R_rad = params.get_matrices()
    
    for tr in tracks:
        #print('predtr.X)
        X_pred = np.dot(F,tr.X)        # x_{k+1} = F x_{k} + q

        P_pred = np.dot(F, tr.P)    # P_{k+1} = F P_{k} F^T + Q
        P_pred = np.dot(P_pred, F.T)
        P_pred = np.add(P_pred, Q)

        tr.predict(X_pred, P_pred)
        #print(tr.X)
    
    
    return tracks

def associate(tracks, detections):

    dt, gate, lamb, max_invisible = params.get_params()

    try:
        F,H,Q,Rinv,R_cam,R_rad = params.get_matrices()
    except TypeError:
        print('TYPERRROR in associate')
        return (tracks, detections, [], [])
    R = Rinv
    distances = [(0,0,0.0) for x in range(0,(len(tracks)*len(detections)))]
    i = 0
    k = 0
    for tr in tracks:
        j = 0

        S = np.dot(H,tr.P)        # S = H P_{k+1} H^T + R
        S = np.dot(S,H.T)
        S = np.add(S,R)

        K = np.dot(tr.P,H.T)        # K = P_{k+1} H^T S^-1
        try:
            K = np.dot(K,np.linalg.inv(S))
        except np.linalg.LinAlgError as e:
            print("K ", K)
            print("S ", S)
            print('\n\n')

        tr.set_KS(K, S)

        for det in range(detections.shape[0]):
            a = np.zeros(6)
            a[0:4] = detections[det,1:5]
            b = np.zeros(6)
            b[0:4]= detections[det,5:9]
            op = Rinv @ (R_rad @ b + R_cam @ a)
            R = Rinv
            
            v = op - np.dot(H, tr.X)    # v = z - H x
            d = np.dot(v.T,np.linalg.inv(tr.S))    # d = v^T S^-1 v
            
            d = np.dot(d,v)
            
            d = max(0.0,d)
            """ 
            Ligne en dessous: check class sert a prendre en compte la classe 
            en information supp pour les distances
            """
            #d += check_classe(det.get_class(), tr.classe)

            #if not tr.multi:# and len(detections) == 1:
            #    d -= 0.8
            #    d = max(0.0,d)

            distances[k] = (i, j, d)
            k += 1

            j += 1
        i += 1

    distances.sort(key=lambda tup: tup[2])
    
    m = []
    n = []

    for elem in distances:
        if(elem[2] < gate):
            m.append(elem)
        else:
            #print(elem[2])
            #print(tracks[elem[0]].classe)
            #print(detections[elem[1]].c0)
            n.append(elem)
            
    l1 = []
    l2 = []
    associated = []
    for e in m:
        if not e[1] in l1 and (not e[0] in l2 or not associate_det_one):
            l1.append(e[1])
            l2.append(e[0])
            associated.append(e)

    new = []
    for e in n:
        if not e[1] in l1:
            l1.append(e[1])
            new.append(e)

    return (tracks, detections, associated, new)

def update(tracks, detections, associated, new):

    dt, gate, lamb, max_invisible = params.get_params()

    try:
        F,H,Q,Rinv,R_cam,R_rad = params.get_matrices()
    except TypeError:
        print('TypeError in update')
        return  (tracks, new, detections)

    sum_p = np.array(len(tracks)*[0.0])
    for e in associated: 
        sum_p[e[0]] += math.exp(-0.5*e[2])    # sum_{l=1}^{m'} exp(0.5*(d{i,l})^2)
        
    prob = []
    # classes = [] 
    # for tr in tracks:
    #     classes.append((0.0, tr.classe))
        
    prob_0 = [(x,0,1.0) for x in range(0,len(tracks))] 
    for e in associated:
        sum_i = sum_p[e[0]]
        #classes[e[0]] = get_max_class(classes[e[0]], detections[e[1]])
        if(sum_i != 0.0):
            p_d = 0.95 #detections[e[1]].get_confidence()
            """
            indice de confiance pour detection de véhicule camera
            """
            S = tracks[e[0]].S
            k = (lamb)*((1-p_d)/p_d)*math.sqrt(np.linalg.det(S))
            p = math.exp(-0.5*e[2])/(k+sum_i)
            prob.append((e[0],e[1],p))

            p_0 = k/(k+sum_i)
            prob_0[e[0]] = ((e[0], -1, p_0))
        else:
            prob_0[e[0]] = ((e[0], -1, 1.0))


    s_pv = np.array(len(tracks)*[6*[0.0]])
    
    for p in prob:
        a0 = np.zeros(6)
        a0[0:4] = detections[p[1],1:5]
        b0 = np.zeros(6)
        b0[0:4]= detections[p[1],5:9]
        #op= R_rad @ Rinv @ a0 + R_cam @ Rinv @ b0
        op = Rinv @ (R_rad @ b0 + R_cam @ a0)
        
        b = (op - np.dot(H, tracks[p[0]].X))
        s_pv[p[0]] = np.add(s_pv[p[0]], p[2]*b)     # sum_{}^{} p_{i,j}v_{i,j}

    s_pvt = len(tracks)*[6*[0.0]]
    for p in prob:
        a0 = np.zeros(6)
        a0[0:4] = detections[p[1],1:5]
        b0 = np.zeros(6)
        b0[0:4]= detections[p[1],5:9]
        #op= R_rad @ Rinv @ a0 + R_cam @ Rinv @ b0
        op = Rinv @ (R_rad @ b0 + R_cam @ a0)
        a = np.dot((op - np.dot(H, tracks[p[0]].X)), (op - np.dot(H, tracks[p[0]].X)).T)
        s_pvt[p[0]] = np.add(s_pvt[p[0]], p[2]*a)     # sum_{}^{} p_{i,j}v_{i,j}v_{i,j}^T

    U = np.array(len(tracks)*[0.0])
    i = 0
    for p0 in prob_0:
        u = np.dot(s_pv[i],s_pv[i].T)     # U_i = (1 - p_{i,0})*(sum*sum^T)
        u = (1-p0[2])*u
        U[i] = u

        i += 1

    i = 0
    for tr in tracks:
        a = np.dot(tr.K,tr.S)
        a = np.dot(a,(tr.K).T)
        a = (1 - prob_0[i][2])*a
        
        b = np.dot(tr.K,(s_pvt[i] - U[i])*(tr.K).T)

        P = tr.P - a + b

        X = tr.X + np.dot(tr.K,s_pv[i])
    
        found = False
        if(sum_p[i] != 0.0):
            found = True

        tr.update(X, P, found)
        #print(X,P)
        i += 1

    #for elem in associated:
     #   tracks[elem[0]].update_sensor(detections[elem[1]].get_sensor())


    return (tracks, new, detections)

def drawing(res,ground_,figures):
    """ results drawing"""
    x = np.zeros(res.shape[0]//6)
    y = np.zeros(res.shape[0]//6)
    z = np.zeros(res.shape[0]//6)
    for i in range(len(res)//6 ):
        x[i] = res[(i*6)] * np.sin(res[i*6 +1] )* np.cos(res[i*6+2])
        y[i] = res[(i*6)] * np.sin(res[i*6 +1] )* np.sin(res[i*6+2])
        z[i] = res[(i*6)] * np.cos(res[i*6+1])
    dist = np.sqrt(x**2 + y**2+ z**2)
    fig = figures[0]
    plt.scatter(x,y,c = 'blue')
    plt.xlabel('x in [m]')
    plt.ylabel('y in [m]')
    plt.title('trajectoire Kalman vs réelle trajectoire')
    plt.scatter(x_true,y_true,c= 'red')
    plt.legend(['traj estimated','traj réele'])
"simulation"
# def getcarte(res):
#     x = res[0] * np.sin(res[1] )* np.cos(res[2])
#     y = res[(0)] * np.sin(res[1] )* np.sin(res[2])
#     z = res[0]*np.cos(res[1])
#     return x,y,z
"cas réel"
def getcarte(res):
    x = res[0] * np.sin(res[1] *np.pi/180)* np.cos(res[2]*np.pi/180)
    y = res[(0)] * np.sin(res[1] *np.pi/180)* np.sin(res[2]*np.pi/180)
    z = res[0]*np.cos(res[1]*np.pi/180)
    return x,y,z



data_esti = 'data_est_cam00.csv'
data = 'data_cam00.csv'
#data_esti = 'data_final_ok.csv'
aA = pd.read_csv(data_esti,sep=';',header = None).values
parameters = [1]
Lister = []
maxim = []
for o in parameters:
        params = kalman_params()
                #min_dist_gate = o
        #max_invisible_no_obs = 
        #min_det_obstruction = o #nope
        #max_invisible_obs = o #200np.linspace(0,20,20)
        # min_obs_front = 0 #influence pas
        # delta_time = 1/30 
        #min_dist_gate = o#0.28 #influence
        #max_invisible_no_obs
        #lambda_fp = 0 #math.sqrt(2*math.pi)*4.0 #parametre useless
        #max_invisible_no_obs = o#influence np.linspace(0,20,20)
        #new_thresh = o
        timer = Timer()
        timer.start()
        L,maxi =    kalman(data_esti,data)
        timer.stop()
        print(timer.interval)
        # Lister.append(L)
        # maxim.append(maxi)
x_cam = aA[:,1]*np.sin(aA[:,2]*np.pi/180)*np.cos(aA[:,3]*np.pi/180)
y_cam = aA[:,1]*np.sin(aA[:,2]*np.pi/180)*np.sin(aA[:,3]*np.pi/180)
x_rad = aA[:,5]*np.sin(aA[:,6]*np.pi/180)*np.cos(aA[:,7]*np.pi/180)
y_rad = aA[:,5]*np.sin(aA[:,6]*np.pi/180)*np.sin(aA[:,7]*np.pi/180)

# plt.figure()
# plt.scatter(x_cam,y_cam,c = 'blue')
# plt.scatter(x_rad,y_rad,c = 'red')
# plt.plot(parameters,Lister)
# # # #plt.plot(parameters,maxim)
# plt.title('MSE par âge maximale de piste non associée ')
# plt.xlabel('âge maximale de piste non associée')
# plt.ylabel('MSE')
# # vrai = pd.read_csv(data,sep=';',header=None).values
# Lister.append(kalman(data_esti,data))
# plt.plot(parameters,Lister)
# plt.title('MSE par age maximale de la piste non associée')
# plt.xlabel('age maximale de la piste non  associée')
# plt.ylabel('MSE')
# vrai = pd.read_csv(data,sep=';',header=None).values
# esti = pd.read_csv(data_esti,sep=';',header=None).values
# data_cam = transform(esti[:,1:5])
# data_rad = transform(esti[:,5::])
# truth = transform(vrai[:,1::])
# x = data_cam[:,0] * np.sin(data_cam[:,1] )* np.cos(data_cam[:,2])
# y = data_cam[:,0] * np.sin(data_cam[:,1] )* np.sin(data_cam[:,2])
# z = data_cam[:,0]*np.cos(data_cam[:,1])
# data_cam0 = np.array([x,y,z]).T
# plt.figure()
# for i in range(211,243):
    
#     res = r[int(i),:]
#     x,y,z = getcarte(res[1::])
#     plt.scatter(x,y)
    
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        