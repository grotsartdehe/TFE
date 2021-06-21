
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
    Xpos2D = df['2D_XPos'].values
    Ypos2D = df['2D_YPos'].values
    pitch =  -pos_cam[5]*np.pi/180
    
    yaw = -pos_cam[6]*np.pi/180
    Rz =np.array( [[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    Ry = np.array([[np.cos(pitch),0, np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    R = Rz@Ry
    
    #Posnew =  R @  np.array([Xpos1,Ypos1,Zpos1])
    Posnew =  np.array([Xpos1,Ypos1,Zpos1])
    Xpos = Posnew[0,:]
    Ypos = Posnew[1,:]
    Zpos = Posnew[2,:]
    d = np.sqrt(Xpos**2 + Ypos**2 + Zpos**2)
    
    #normaliser pour obtenir vecteur cam-vehicule
    Xposdir = Xpos/d
    Yposdir = Ypos/d
    Zposdir = Zpos/d
    W = 1920
    H = 1280
    cond2 = (Xpos2D >= 0) & (Xpos2D <= W) & (Ypos2D >= 0) & (Ypos2D <= H)
    Xpos2D = Xpos2D[cond2]
    Ypos2D = Ypos2D[cond2]
    cond = (d < 70) * cond2
    
    
    v = df['Vel']/100
    
    Xdir = df['XDir']
    Ydir = df['YDir']
    Zdir= df['ZDir']
    
    #Dirnew =  R @  np.array([Xdir,Ydir,Zdir])
    Dirnew =  np.array([Xdir,Ydir,Zdir])
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
    #store = np.array([d[cond],theta[cond]*180/np.pi,phi[cond]*180/np.pi,v1[cond]]).T
    #print(store)
   
    
    return d[cond],v1[cond],theta[cond],phi[cond],classcar[cond],xsi[cond],v[cond]#+pi/2
    
    
    
def CreateandSearch(FX_csv,pos_cam,cam_number):
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
    d_real,v_real,theta,phi,classcar,xsi,vabs = extract(data,pos_cam)
    v_real = v_real.values
    print("vitesse", v_real)
    """Generation et recherche dsitance, vitessede heatmap d,v"""
    Zdv,Za = RadarGen(classcar.values,d_real,v_real,theta,phi,xsi,vabs.values)
    #plotDV(Zdv)
    d_esti,v_esti,lignes, colonnes = Searchdv(Zdv,256,256)
    
    
    

    """Generation et recherche des angles theta, phi limité dans l'espace 
    d'ambiguité"""
    index = Za[lignes,colonnes] -2
    # plotDV(Zdv)
    # plotDV(Za)
    if len(d_esti)==0:
        return []
    theta_esti = np.array(np.zeros((d_esti.shape)))
    
    phi_esti = np.array(np.zeros((d_esti.shape)))
    count = 0
    index = np.int_(index)
    for m in index:
        if m ==-1:
            Z= np.random.normal((256,256))

            theta_esti[count],phi_esti[count] = Searchangle(Z,cam_number )
        else:
            
            Z = ambiguite(theta[m],phi[m],cam_number=cam_number)
            #plotAngles(Z)
            theta_esti[count],phi_esti[count] = Searchangle(Z,cam_number=cam_number )
            #print('true angles',theta[m]*180/pi,phi[m]*180/pi)
            #print('false angles',theta_esti[count]*180/pi,phi_esti[count]*180/pi)
            vect2 = [0,theta[m]*180/pi,phi[m]*180/pi,0,theta_esti[count]*180/pi,phi_esti[count]*180/pi,0]
            vect2 = correctionAngle(vect2)
            #print(vect2)
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
    cam_number = int( csv_folder[-1])
    pos_cam = os.path.join(csv_folder,'pos_cam_'+csv_folder[-2:]+'.csv')
    df = pd.read_csv(pos_cam, sep =';')
    
    pos_cam = df.values[1,:]#[df.iloc[2]['Xpos'],df.iloc[2]['Ypos'],df.iloc[2]['Zpos']]
    Thelist = []
    csv_data = os.listdir(csv_folder)
    csv_data.sort()
    counter = 581
    for i in csv_data:
        if  not i.startswith('.~lock') and not i.startswith('pos') and not i.endswith('.jpg'):
        
            n = 1150
            if  counter == n  or( counter > n and counter<(n)) :
                 
                file = os.path.join(csv_folder,i)
                print(file)
                df = pd.read_csv(file,sep =';',index_col=False )
                v_abs = df['Vel']/100
                
                d_real,v_real,theta,phi,classcar,xsi,vabs = extract(df,pos_cam)
                v_abs = df['Vel']/100 
                v_abs = v_abs[v_real.index].values
                
                test = CreateandSearch(file,pos_cam,cam_number)
                print(test)
                if len(test)==0:
                    print('vecteur est vide')
                    continue
                if test.shape[0]>0:
                    min_dist = np.ones((test.shape[0]))*500
                    index = np.zeros((test.shape[0]))
                    
                    for i in range(test.shape[0]):
                        for j in range(len(d_real)):

                            l = (test[i,0] - d_real[j])**2 #+ (v_esti[i] - v_real[j])**2

                            if min_dist[i] >l:
                                min_dist[i] = l 
                                index[i] = j
                
                table = np.zeros((test.shape[0],7))
                newtab= np.zeros((test.shape[0],7))
                tab0= np.zeros((test.shape[0],7))
                count= 0
                index = np.int_(index)
                v_real = v_real.values
                
                for m in index:
                    
                    table[count,0]=d_real[m]
                    table[count,1]=theta[m]*180/np.pi
                    table[count,2]=phi[m]*180/np.pi
                    #table[count,3]=v_abs[m]
                    table[count,3::]=test[count,:]
                    table[count,6]=v_real[m]
                    #print('vrad',v_real[m])
                    count+=1 
                #print('detection',len(index))
                
                for i in range(table.shape[0]):
                    
                    newtab[i,:] = correction(tab0[i,:],table[i,:])
                #print(newtab)
                # if counter > n:
                #     newtab= correctionvitesse(tab0,newtab)
                #        #print('boucle')
                
                tab0 = newtab
                #print(newtab)
            counter += 1
                
                
                
"""                if not counter %1322:
                    filename = 'bigfile' + str(counter)
                    outfile = open(filename,'wb')
                    pickle.dump(Thelist,outfile)
                    outfile.close()
                    Thelist= []"""
    
    
    
    
