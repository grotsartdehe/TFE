
"""
Created on Fri Jan 29 17:23:17 2021

@author: kdesousa


TEST de FICHIER FX AVEC TRAJECTOIRE
"""
import numpy as np
import pandas as pd
import math
from kalman_tools_demo import track, detection, kalman_params, frame
#from kalman_graphics import kalman_draw
import os
import time
import datetime
#import cv2
import pickle
from meta_params import *
from RadarGen import *
import matplotlib.pyplot as plt
params = kalman_params()

def kalman(folder):


    dt, gate, lamb, max_invisible = params.get_params()
    count = 1820
    name = os.path.join(folder,'00_'+str(count).zfill(8)+'.csv')
    tracks = []
    i =0
    poscam = pd.read_csv(os.path.join(folder,'pos_cam_00.csv'),sep =';').values
    x0 = poscam[1,1]
    y0 = poscam[1,2]
    z0 = poscam[1,3]
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    while(os.path.isfile(name) == 1) and count <1860:
        print('counter :' ,count)
        name = os.path.join(folder,'00_'+str(count).zfill(8)+'.csv')
        frame = load_data(folder,name, dt)

        if i ==0:
            
            tracks = kalman_estimate(tracks, frame.get_detections())
            i=1
        else: tracks = kalman_estimate(tracks, frame.get_detections())
        
        
        """if len(tracks) > 0:
            print(tracks[0].X)"""
        
            #kalman_draw(tracks)
        count +=1
        data = pd.read_csv(name,sep =';')
        d= data.values
        x = np.array( d[:,3],dtype=int) -x0
        y = np.array( d[:,4],dtype=int) - y0
        z = np.array( d[:,5],dtype=int) -z0
        """print('x =',x/100)
        print('y =',y/100)
        print('z =',z/100)"""
        classe = np.array( d[:,2],dtype=int)
        v = np.array( d[:,8],dtype=int)/100
        dist = np.sqrt(x**2 + y**2+ z**2)/100
        theta = np.arccos(z/(100*dist))
        phi = np.arctan2(y,x)
        
        
        k = dist<70 
        m= dist>20
        k = k*m
        """print(tracks[0].X[0],tracks[0].X[1],tracks[0].X[2])
        print(dist[k],theta[k],phi[k])"""
        result = []
        for mas in tracks:
            result = np.append(result,[mas.get_vectorX()])
        
        
        
        drawing(result,x[k]/100,y[k]/100,z[k]/100,v[k],[fig1,fig2,fig3])
    return tracks

def drawing(res,x_true,y_true,z_true,v,figures):
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
    
    """fig = figures[1]
    plt.plot(res[3],c = 'blue')
    plt.xlabel('d in [m/s]')
    plt.ylabel('v in [m/s]')
    plt.title('d,v Kalman vs réelle ')
    plt.plot(v,c= 'red')
    plt.legend([' estimée',' réele'])
    """
    
    
    """pitch = np.radians(-12.999)

    yaw = np.radians(-2.5)
    Rz = np.array([[np.cos(-yaw),-np.sin(-yaw),0],[np.sin(-yaw),np.cos(-yaw),0],[0,0,1]])
    Ry = np.array([[np.cos(-pitch),0,np.sin(-pitch)],[0,1,0],[-np.sin(-pitch),0,np.cos(-pitch)]])
    R = Rz@Ry
    
    new_x_true = np.zeros(x_true.shape[0])
    new_y_true = np.zeros(x_true.shape[0])
    new_z_true = np.zeros(x_true.shape[0])
    
    for l in range(x_true.shape[0]):
        [[new_x_true],[new_y_true],[new_z_true]] = np.array([[x_true[l]],[y_true[l]],[z_true[l]]])"""
    
    
def load_data(folder,name,count,dt=1/30):
    """
    Parameters
    ----------
    radar_folder : string
        emplacement des preditctions du Search Algo
    cam_folder : TYPE
        emplacement des prédictions de yolo
    dt : float
        intervalle de temps entre 2 instances
    Returns
    -------
    list de frame de chaque instance
    """
    # class x,y, width, height,d,v
    
    
    #totradar = len(os.listdir(radar_folder))
    #totcam = len(os.listdir(cam_folder))
    j=0
    """fold = os.listdir(folder)
    fold.sort()"""
    poscam = pd.read_csv(os.path.join(folder,'pos_cam_00.csv'),sep =';').values
    x0 = poscam[1,1]
    y0 = poscam[1,2]
    z0 = poscam[1,3]
    pitch = poscam[1,5]
    yaw = poscam[1,6]

    fra = frame()
    data = pd.read_csv(name,sep =';')
    if len(data) ==0  :
        fra.set_time(count)
        
    else: 
        d= data.values
        x = np.array( d[:,3],dtype=int) -x0
        y = np.array( d[:,4],dtype=int) - y0
        z = np.array( d[:,5],dtype=int) -z0
        classe = np.array( d[:,2],dtype=int)
        v = np.array( d[:,8],dtype=int)/100 +np.random.normal()
        dist = np.sqrt(x**2 + y**2+ z**2)/100 
        theta = np.arccos(z/(100*dist)) 
        phi = np.arctan2(y,x)
        
        for k in range(d.shape[0]):
            # (self,classcar,d,v,theta,phi,p,timestep)
            if dist[k]<80:
                detec = detection(classe[k],dist[k],v[k],theta[k],phi[k],1,j)
                #print(detec.get_X())
                fra.add_detection(detec)
                fra.set_time(count)
                j+=1
            #print(fra.get_detections()[0].get_X())
            
        
        
        

    return fra

def kalman_estimate(tracks, detections):
    
    # Si il n'y a aucune piste, passer l'assoc et l'update 
    if(len(tracks) == 0):
       
        return housekeep(tracks, detections)
    
    # Prédiction de l'état suivant    
    tracks = predict(tracks)
    

    # Si pas de détection, passer l'assoc
    if(len(detections) == 0):
        print('detections==0')
        for tr in tracks:
            tr.invisible += 1
        return housekeep(tracks, detections)

    # Association 
    tracks, detections, associated, new = associate(tracks, detections)
    
    # Update
    tracks, new, detections = update(tracks, detections, associated, new)
    
    # Housekeep
    filtered_tracks = housekeep(tracks, detections, new)
    
        
    return filtered_tracks

def associate(tracks, detections):

    dt, gate, lamb, max_invisible = params.get_params()

    try:
        F, H, Q, R = params.get_matrices() # toutes les detections d'une frames sont du même capteur

    except TypeError:

        return (tracks, detections, [], [])
        
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

        for det in detections:

            v = det.get_X() - np.dot(H, tr.X)    # v = z - H x
            
            d = np.dot(v.T,np.linalg.inv(tr.S))    # d = v^T S^-1 v
            
            d = np.dot(d,v)
            
            d = max(0.0,d)
           
            d += check_classe(det.get_class(), tr.classe)

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

def predict(tracks):

    
    F, H, Q, R = params.get_matrices()

    for tr in tracks:

        X_pred = np.dot(F,tr.X)        # x_{k+1} = F x_{k} + q
        
        P_pred = np.dot(F, tr.P)    # P_{k+1} = F P_{k} F^T + Q
        P_pred = np.dot(P_pred, F.T)
        P_pred = np.add(P_pred, Q)
        
        tr.predict(X_pred, P_pred)

    return tracks
            
def update(tracks, detections, associated, new):

    dt, gate, lamb, max_invisible = params.get_params()

    try:
        F, H, Q, R = params.get_matrices() # toutes les detections d'une frames sont du même capteur

    except TypeError:

        return  (tracks, new, detections)

    sum_p = np.array(len(tracks)*[0.0])
    for e in associated: 
        sum_p[e[0]] += math.exp(-0.5*e[2])    # sum_{l=1}^{m'} exp(0.5*(d{i,l})^2)

    prob = []
    classes = [] 
    for tr in tracks:
        classes.append((0.0, tr.classe))
        
    prob_0 = [(x,0,1.0) for x in range(0,len(tracks))] 
    for e in associated:
        sum_i = sum_p[e[0]]
        
        classes[e[0]] = 0 # get_max_class(classes[e[0]], detections[e[1]])
        if(sum_i != 0.0):
            p_d = detections[e[1]].get_confidence()
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
        b = (detections[p[1]].get_X() - np.dot(H, tracks[p[0]].X))
        s_pv[p[0]] = np.add(s_pv[p[0]], p[2]*b)     # sum_{}^{} p_{i,j}v_{i,j}

    s_pvt = len(tracks)*[6*[0.0]]
    for p in prob:
        a = np.dot((detections[p[1]].get_X() - np.dot(H, tracks[p[0]].X)), (detections[p[1]].get_X() - np.dot(H, tracks[p[0]].X)).T)
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
        
        #tr.update(X, P, classes[i][1], found)
        tr.update(X, P, found)
        i += 1

     #useless dans notre cas
    """for elem in associated:
        tracks[elem[0]].update_sensor(detections[elem[1]].get_sensor())"""


    return (tracks, new, detections)


def housekeep(tracks, detections, new = 0):
    
    # Create new tracks with unassociated detections
    
    if not new == 0 and len(new) > 0:

        #new_thresh = min_distance_new
        new_tracks = []
        for n in new:

            det = detections[n[1]]

            new_thresh = new_thresh_classes[det.get_class()]

            if(n[2] > new_thresh):

                new_t = track(0)
                F, H, Q, R = params.get_matrices()
                R_0 = params.get_init_matrice()
                new_t.create(det.get_X(), det.get_class(), R_0)
                new_tracks.append(new_t)

        # Keep only good tracks
        
        tracks.extend(new_tracks)
        """ START INIT """
    if(len(tracks) == 0 and len(detections) != 0):
        
        new_tracks = []

        for det in detections:
            
            if det is not None:
                new_t = track(0)
                try:
                    F, H, Q, R = params.get_matrices()
                    
                except:    
                    
                    F, H, Q, R = params.get_matrices()
                
                new_t.create(det.get_X(), det.get_class(), R)
                new_tracks.append(new_t)
                
        
        tracks.extend(new_tracks)
        
    dt, gate, lamb, max_invisible = params.get_params()

    keep_tracks = []
    for tr in tracks:
        
        #thresh = obstruction_thresh

        obs = False

        for t in tracks:
            thresh = 0.8*math.degrees(math.atan(objects_obs_size[t.classe]/(2*t.X[2])))
            
            if(abs(tr.X[0] - t.X[0]) < thresh and (tr.X[2] > t.X[2]) and tr.min_det > min_det_obstruction):
                if(t.min_det > min_obs_front):
                    
                    tr.obstruction += 1
                    obs = True

        if(obs == False):
            tr.obstruction = 0
    
            #else:
            #    tr.obstruction = 0

            #tr.obstruction = max(0, tr.obstruction)

        if(tr.invisible < max_invisible_obs):
            if((tr.invisible < max_invisible_no_obs) or (tr.obstruction > 0)):
                keep_tracks.append(tr)    
                
    return keep_tracks



def check_classe(d_c, t_c):


    high = 0.7
    middle = 0.6
    low = 0.05

    if(t_c == "init"):
        return 10.0

    dic_car = {"car":0.0, "person":high, "truck":high, "suv":low, "van":low, "bicycle":high, "motorbike":high, "bus":high, "scooter":high, "ucl":middle, "trailer":high, "dog":high, "petit":middle, "moyen":0.0, "grand":middle}

    dic_perso = {"car":high, "person":0.0, "truck":high, "suv":high, "van":high, "bicycle":low, "motorbike":low, "bus":high, "scooter":low, "ucl":middle, "trailer":high, "dog":high, "petit":0.0, "moyen":middle, "grand":high}

    dic_truck = {"car":middle, "person":high, "truck":0.0, "suv":middle, "van":middle, "bicycle":high, "motorbike":high, "bus":middle, "scooter":high, "ucl":middle, "trailer":middle, "dog":high, "petit":high, "moyen":middle, "grand":0.0}

    dic_suv = {"car":low, "person":high, "truck":middle, "suv":0.0, "van":middle, "bicycle":high, "motorbike":high, "bus":high, "scooter":high, "ucl":high, "trailer":middle, "dog":high, "petit":middle, "moyen":0.0, "grand":middle}

    dic_van = {"car":low, "person":high, "truck":middle, "suv":middle, "van":0.0, "bicycle":high, "motorbike":high, "bus":high, "scooter":high, "ucl":high, "trailer":high, "dog":high, "petit":middle, "moyen":0.0, "grand":middle}

    dic_bic = {"car":high, "person":middle, "truck":high, "suv":high, "van":high, "bicycle":0.0, "motorbike":low, "bus":high, "scooter":low, "ucl":high, "trailer":high, "dog":low, "petit":0.0, "moyen":middle, "grand":high}

    dic_moto = {"car":high, "person":middle, "truck":high, "suv":high, "van":high, "bicycle":low, "motorbike":0.0, "bus":high, "scooter":low, "ucl":high, "trailer":high, "dog":low, "petit":0.0, "moyen":middle, "grand":high}

    dic_bus = {"car":high, "person":high, "truck":middle, "suv":high, "van":high, "bicycle":high, "motorbike":high, "bus":0.0, "scooter":high, "ucl":high, "trailer":high, "dog":high, "petit":high, "moyen":middle, "grand":0.0}

    dic_scooter = {"car":high, "person":middle, "truck":high, "suv":high, "van":high, "bicycle":low, "motorbike":low, "bus":high, "scooter":0.0, "ucl":high, "trailer":high, "dog":high, "petit":0.0, "moyen":middle, "grand":high}

    dic_ucl = {"car":middle, "person":high, "truck":middle, "suv":middle, "van":high, "bicycle":high, "motorbike":high, "bus":high, "scooter":high, "ucl":0.0, "trailer":high, "dog":high, "petit":middle, "moyen":0.0, "grand":high}

    dic_trail = {"car":middle, "person":high, "truck":middle, "suv":high, "van":high, "bicycle":high, "motorbike":high, "bus":high, "scooter":high, "ucl":high, "trailer":0.0, "dog":high, "petit":middle, "moyen":0.0, "grand":middle}

    dic_dog = {"car":high, "person":middle, "truck":high, "suv":high, "van":high, "bicycle":low, "motorbike":middle, "bus":high, "scooter":middle, "ucl":high, "trailer":high, "dog":0.0, "petit":0.0, "moyen":middle, "grand":high}

    dic_petit = {"car":low, "person":0.0, "truck":high, "suv":middle, "van":middle, "bicycle":0.0, "motorbike":0.0, "bus":high, "scooter":0.0, "ucl":low, "trailer":high, "dog":0.0, "petit":0.0, "moyen":middle, "grand":high}

    dic_moyen = {"car":0.0, "person":middle, "truck":middle, "suv":0.0, "van":0.0, "bicycle":middle, "motorbike":middle, "bus":high, "scooter":middle, "ucl":low, "trailer":0.0, "dog":middle, "petit":middle, "moyen":0.0, "grand":middle}

    dic_grand = {"car":middle, "person":high, "truck":0.0, "suv":middle, "van":middle, "bicycle":high, "motorbike":high, "bus":0.0, "scooter":high, "ucl":middle, "trailer":middle, "dog":high, "petit":high, "moyen":middle, "grand":0.0}

    dico = {"car":dic_car, "person":dic_perso, "truck":dic_truck, "suv":dic_suv, "van":dic_van, "bicycle":dic_bic, "motorbike":dic_moto, "bus":dic_bus, "scooter":dic_scooter, "ucl":dic_ucl, "trailer":dic_trail, "dog":dic_dog, "petit":dic_petit, "moyen":dic_moyen, "grand":dic_grand}

    malus = dico[t_c][d_c]

    return high 



if __name__ == '__main__':

    radar_folder = "/home/kdesousa/Documents/GitHub/TFE/Kalman/Data/radar/data-yolo-2"
    camera_folder = "/home/kdesousa/Documents/GitHub/TFE/Kalman/Data/cam/data-yolo-2"
    folder = '/home/kdesousa/Documents/GitHub/TFE/Kalman/2021_03_31_15_11_10_392 - Copy/cam_00'
    kalman(folder)
