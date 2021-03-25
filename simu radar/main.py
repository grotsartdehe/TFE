
"""
Created on Sun Mar  7 18:35:07 2021

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
    cond = d < 80
    
    v = df['Vel']*3.6/100
    
    Xdir = df['XDir']
    Ydir = df['YDir']
    
    #projection orthogonale
    norm =  np.sqrt(Xposdir**2 + Yposdir**2)
    Vdir = (Xdir*Xposdir + Ydir*Yposdir)/norm # diviser par norm = 1
    
    v = v*Vdir
    
    
    
    
    phi = np.arccos(Zpos/d)
    theta = np.arctan2(Ypos,Xpos)
    classcar = df['Cat']
    return d[cond],v[cond],theta[cond],phi[cond],classcar[cond]
    
    
    


        
        
        
        
if __name__ == '__main__':
    csv_folder= '/home/kdesousa/Documents/GitHub/TFE/simu radar/2021_03_09_17_19_58_165/cam_01'
    pos_cam = os.path.join(csv_folder,'pos_cam_01.csv')
    df = pd.read_csv(pos_cam, sep =';')
    pos_cam = [df.iloc[2]['Xpos'],df.iloc[2]['Ypos'],df.iloc[2]['Zpos']]
    Thelist = []
    csv_data = os.listdir(csv_folder)
    csv_data.sort()
    counter = 0
    for i in csv_data:
        if  not i.startswith('.~lock') and not i.startswith('pos') and not i.endswith('.jpg'):
            heat = heatmap(counter)
            counter += 1
            
            if  counter == 1322 or counter == 750: 
                 
                file = os.path.join(csv_folder,i)
                
                dataf = pd.read_csv(file,sep =';')
                
                d,v,theta,phi,classcar = extract(dataf,pos_cam)
                
                Zdv, Zangle = RadarGen(classcar,d,v,theta,phi)
                heat.store(Zdv,Zangle)
                Thelist.append(heat)
                if not counter %1322:
                    filename = 'bigfile' + str(counter)
                    outfile = open(filename,'wb')
                    pickle.dump(Thelist,outfile)
                    outfile.close()
                    Thelist= []
    
    
    
    
