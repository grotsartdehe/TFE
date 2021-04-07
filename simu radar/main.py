
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
    phi : angle d'élevation
    """
    Xpos = (df['XPos']-pos_cam[0])/100
    Ypos = (df['YPos']-pos_cam[1])/100
    Zpos = (df['ZPos']-pos_cam[2])/100
    Xdir = df['XPos']
    d = np.sqrt(Xpos**2 + Ypos**2 + Zpos**2)
    
    #normaliser pour obtenir vecteur cam-vehicule
    Xposdir = Xpos/d
    Yposdir = Ypos/d
    #Zposdir = Zpos/d
    cond = d < 70
    
    v = df['Vel']/100
    
    Xdir = df['XDir']
    Ydir = df['YDir']
    
    #projection orthogonale
    norm =  np.sqrt(Xposdir**2 + Yposdir**2)
    Vdir = (Xdir*Xposdir + Ydir*Yposdir)/norm # diviser par norm = 1
    
    v = v*Vdir
    
    
    
    
    theta = np.arccos(Zpos/d)
    phi = np.arctan2(Ypos,Xpos)
    classcar = df['Cat']
    return d[cond],v[cond],theta[cond],phi[cond],classcar[cond]
    
    
    
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
    d_real,v_real,theta,phi,classcar = extract(data,pos_cam)
   
    """Generation et recherche dsitance, vitessede heatmap d,v"""
    Zdv = RadarGen(classcar,d_real,v_real,theta,phi)
    d_esti,v_esti = Searchdv(Zdv,256,256)
    # plotDV(Zdv)
    
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
            for j in range(len(d_real.values)):
                    
                    
                    
                    l = (d_esti[i] - d_real.values[j])**2 + (v_esti[i] - v_real.values[j])**2
                   
                    
                    if min_dist[i] >l:
                        min_dist[i] = l 
                       
                        index[i] = j
    else :
        
        return []
    """Generation et recherche des angles theta, phi limité dans l'espace 
    d'ambiguité"""
    theta_esti = np.array(np.zeros((d_esti.shape)))
    
    phi_esti = np.array(np.zeros((d_esti.shape)))
    count = 0
    index = np.int_(index)
    for m in index:
        
        Z = ambiguite(theta.values[m],phi.values[m])
        theta_esti[count],phi_esti[count] = Searchangle(Z )
        
        count +=1
    # print('real phi',phi.values*180/pi)
    # print('estim phi',phi_esti*180/pi)
    # print('real theta',theta.values*180/pi)
    # print('estim theta',theta_esti*180/pi)
    """creation de la liste"""
    lister = np.zeros((d_esti.shape[0],4))
    for i in range(len(d_esti)):
        lister[i,:]=d_esti[i],theta_esti[i],phi_esti[i],v_esti[i]
    # print(lister)
    return lister
        
        
if __name__ == '__main__':
    csv_folder= '/home/kdesousa/Documents/GitHub/TFE/Kalman/2021_04_06_15_40_39_604/cam_00'
    pos_cam = os.path.join(csv_folder,'pos_cam_00.csv')
    df = pd.read_csv(pos_cam, sep =';')
    
    pos_cam = [df.iloc[2]['Xpos'],df.iloc[2]['Ypos'],df.iloc[2]['Zpos']]
    Thelist = []
    csv_data = os.listdir(csv_folder)
    csv_data.sort()
    counter = 581
    for i in csv_data:
        if  not i.startswith('.~lock') and not i.startswith('pos') and not i.endswith('.jpg'):
            heat = heatmap(counter)
            
            
            if  counter == 1204 or counter == 881 or counter == 582: #or counter == 4520: 
                 
                file = os.path.join(csv_folder,i)
                print(file)
                test = CreateandSearch(file,pos_cam)
            counter += 1
                
                 

                
"""                if not counter %1322:
                    filename = 'bigfile' + str(counter)
                    outfile = open(filename,'wb')
                    pickle.dump(Thelist,outfile)
                    outfile.close()
                    Thelist= []"""
    
    
    
    
