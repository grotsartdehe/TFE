
"""
Created on Sun Mar  7 18:35:07 2021
Pour generer et rechercher données tu as besoin de 3 fichiers
le main.py (ce fichier), RadarGen.py et Search.py 
Je dois encore mettre le code au propre
@author: kdesousa
"""

from RadarGen import *
import os
import pandas as pd
import numpy as np
from tools import heatmap
from Search import *
from correction import *
import pickle

def extract(df,pos_cam):
    """
    Parameters
    ----------
    dataframe : panda dataframe
    Returns
    -------
    d : liste of distance.
    v : liste de vitesse
    theta : angle azimutale
    phi : angle d'élevationmain.py
    
    """
    Xpos1 = (df['XPos']-pos_cam[1])/100
    Ypos1 = (df['YPos']-pos_cam[2])/100
    Zpos1 = (df['ZPos']-pos_cam[3])/100
    pitch =  pos_cam[5]*np.pi/180
    # print(pos_cam[5])
    # print(pos_cam[6])
    yaw = pos_cam[6]*np.pi/180
    Rz =np.array( [[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    Ry = np.array([[np.cos(pitch),0, np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    R = Rz@Ry
    
    Posnew =  R @  np.array([Xpos1,Ypos1,Zpos1])
    Xpos = Posnew[0,:]
    Ypos = Posnew[1,:]
    Zpos = Posnew[2,:]
    d = np.sqrt(Xpos**2 + Ypos**2 + Zpos**2)
    
    #normaliser pour obtenir vecteur cam-vehicule
    Xposdir = Xpos/d
    Yposdir = Ypos/d
    Zposdir = Zpos/d
    cond = d < 70
    
    v = df['Vel']/100
    
    Xdir = df['XDir']
    Ydir = df['YDir']
    Zdir= df['ZDir']
    Dirnew =  R @  np.array([Xdir,Ydir,Zdir])
    Xdir = Dirnew[0]
    Ydir = Dirnew[1]
    Zdir= Dirnew[2]
    
    #projection orthogonale
    norm =  np.sqrt(Xposdir**2 + Yposdir**2 + Zposdir**2) 
    Vdir = (Xdir*Xposdir + Ydir*Yposdir+Zdir*Zposdir)/norm # diviser par norm = 1
    
    
    #print(np.arccos(Vdir/v)*180/np.pi)
   
    
    theta = np.arccos(Zpos/d) 
    phi = np.arctan2(Ypos,Xpos)
    classcar = df['ID']
    v1 = v*Vdir
    
    xsi = np.arctan2(Ydir,Xdir)
    #print(xsi*180/np.pi)
    store = np.array([d[cond],theta[cond]*180/np.pi,phi[cond]*180/np.pi,v1[cond]]).T
    
    #print(store)
    
   
    
    return d[cond],v1[cond],theta[cond],phi[cond],classcar[cond],xsi[cond]#+pi/2
    
    
    
def CreateandSearch(FX_csv,pos_cam):
    """
    

    Parameters
    ----------
    FX_csv : string
    emplacement ficher csv
    
    pos_cam : list 
    [Xpos, Ypos,Zpos]
        

    Returns
    -------
    list: retourne liste 1 ligne par véhciule 
                [d,theta,phi,v] estimé par algo 
        

    """
    
    """ extractions des données de FX"""
    data = pd.read_csv(FX_csv,sep =';',index_col=False )
    d_real,v_real,theta,phi,classcar,xsi = extract(data,pos_cam)
    v_real = v_real.values
    """Generation et recherche dsitance, vitessede heatmap d,v"""
    Zdv = RadarGen(classcar.values,d_real,v_real,theta,phi,xsi)
    d_esti,v_esti = Searchdv(Zdv,256,256)
    #plotDV(Zdv)
    
    # if not  type(d_esti) == list:
    #     d_esti = np.array([d_esti])
    #     v_esti = np.array([v_esti])
    
    # print("real dist",d_real.values)
    # print("estimé",d_esti)
    # print("real vitesse rad", v_real.values)
    # print("estimé",v_esti)
    
    if len(d_esti)>0:
        min_dist = np.ones((d_esti.size))*500
        index = np.zeros((d_esti.size))
        for i in range(len(d_esti)):
            for j in range(len(d_real)):
                    
                    
                
                    
                    l = (d_esti[i] - d_real[j])**2 #+ (v_esti[i] - v_real[j])**2
                   
                    
                    if min_dist[i] >l:
                        min_dist[i] = l 
                       
                        index[i] = j
            #print(d_esti[i],d_real[int(index[i])])
        
    else :
        
        return []
    """Generation et recherche des angles theta, phi limité dans l'espace 
    d'ambiguité"""
    theta_esti = np.array(np.zeros((d_esti.shape)))
    
    phi_esti = np.array(np.zeros((d_esti.shape)))
    count = 0
    index = np.int_(index)
    for m in index:
        
        Z = ambiguite(theta[m],phi[m])
        #plotAngles(Z)
        theta_esti[count],phi_esti[count] = Searchangle(Z )
        
        count +=1
    # print('real phi',phi*180/pi)
    # print('estim phi',phi_esti*180/pi)
    # print('real theta',theta*180/pi)
    # print('estim theta',theta_esti*180/pi)
    """creation de la liste"""
    lister = np.zeros((d_esti.shape[0],4))
    for i in range(len(d_esti)):
        lister[i,:]=d_esti[i],(theta_esti[i])*180/pi,phi_esti[i]*180/pi,v_esti[i]
    #print(lister)
    return lister
        
        
if __name__ == '__main__':
    csv_folder= '/home/kdesousa/Documents/GitHub/TFE/Kalman/2021_04_06_15_40_39_604/cam_00'
    pos_cam = os.path.join(csv_folder,'pos_cam_00.csv')
    df = pd.read_csv(pos_cam, sep =';')
    
    pos_cam = df.values[1,:]#[df.iloc[2]['Xpos'],df.iloc[2]['Ypos'],df.iloc[2]['Zpos']]
    Thelist = []
    csv_data = os.listdir(csv_folder)
    csv_data.sort()
    counter = 581
    for i in csv_data:
        if  not i.startswith('.~lock') and not i.startswith('pos') and not i.endswith('.jpg'):
        
            n = 1450
            if  counter == n  or( counter > n and counter<(n+5)) :
                 
                file = os.path.join(csv_folder,i)
                print(file)
                df = pd.read_csv(file,sep =';',index_col=False )
                v_abs = df['Vel']/100
                
                d_real,v_real,theta,phi,classcar,xsi = extract(df,pos_cam)
                v_abs = df['Vel']/100 
                v_abs = v_abs[v_real.index].values
                
                test = CreateandSearch(file,pos_cam)
                
                if test.shape[0]>0:
                    min_dist = np.ones((test.shape[0]))*500
                    index = np.zeros((test.shape[0]))
                    
                    for i in range(test.shape[0]):
                        for j in range(len(d_real)):

                            l = (test[i,0] - d_real[j])**2 #+ (v_esti[i] - v_real[j])**2

                            if min_dist[i] >l:
                                min_dist[i] = l 
                                index[i] = j
                
                table = np.zeros((test.shape[0],8))
                newtab= np.zeros((test.shape[0],8))
                count= 0
                index = np.int_(index)
                v_real = v_real.values
                
                for m in index:
                    
                    table[count,0]=d_real[m]
                    table[count,1]=theta[m]*180/np.pi
                    table[count,2]=phi[m]*180/np.pi
                    table[count,3]=v_abs[m]
                    table[count,4::]=test[count,:]
                    table[count,7]=v_real[m]
                    #print('vrad',v_real[m])
                    count+=1 
                #print('detection',len(index))
                for i in range(table.shape[0]):
                    
                    newtab[i,:] = correctionAngle(table[i,:])
                #print(newtab)
                if counter > n:
                    newtab= correctionvitesse(tab0,newtab)
                       #print('boucle')
                
                tab0 = newtab
                #print(newtab)
            counter += 1
                
                
                
"""                if not counter %1322:
                    filename = 'bigfile' + str(counter)
                    outfile = open(filename,'wb')
                    pickle.dump(Thelist,outfile)
                    outfile.close()
                    Thelist= []"""
    
    
    
    
