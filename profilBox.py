# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 12:49:14 2021

@author: Gauthier_Rotsart
"""
import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
import csv
# def find_ground_truths(df, W=1920, H=1280):
#     df = df.values
#     Aire = np.zeros(len(df[:,0]))
#     max_x = np.zeros(len(df[:,0]))
#     max_y = np.zeros(len(df[:,0]))
#     min_x = np.ones(len(df[:,0]))*W
#     min_y  =np.ones(len(df[:,0]))*H
#     if Aire is None:
#         return np.array([])
#     for j in range(44, 60, 2):
#             x, y = df[:,int(j)], df[:,int(j + 1)]
#             for i in range(len(x)):
#                 if x[i] == -1 or y[i] == -1:
#                     continue
#                 if x[i] < min_x[i]:
#                     min_x[i] = x[i]
                    
#                 if y[i] < min_y[i]:
#                     min_y[i] = y[i]
#                 if x[i] > max_x[i]:
#                     max_x[i] = x[i]
#                 if y[i] > max_y[i]:
#                     max_y[i] = y[i]
#     #cv2.rectangle(frame, (max_x, max_y), (min_x, min_y), (255, 255, 255), thickness)
#     Aire = (max_y-min_y)*(max_x-min_x)
#     return Aire
def find_ground_truths(df, W=1920, H=1280):
    df = df.values
    Aire = np.zeros(len(df[:,0]))
    max_x = np.zeros(len(df[:,0]))
    max_y = np.zeros(len(df[:,0]))
    min_x = np.ones(len(df[:,0]))*W
    min_y  =np.ones(len(df[:,0]))*H
    for j in range(44, 60, 2):
            x, y = df[:,int(j)], df[:,int(j + 1)]
            for i in range(len(x)):
                if x[i] == -1 or y[i] == -1:
                    continue
                if x[i] < min_x[i]:
                    min_x[i] = x[i]
                    
                if y[i] < min_y[i]:
                    min_y[i] = y[i]
                if x[i] > max_x[i]:
                    max_x[i] = x[i]
                if y[i] > max_y[i]:
                    max_y[i] = y[i]
    #cv2.rectangle(frame, (max_x, max_y), (min_x, min_y), (255, 255, 255), thickness)
    Aire = (max_y-min_y)*(max_x-min_x)
    return Aire
count = 800#2000#test de la frame 2437
W = 1920
H = 1080
cm_x = [214]
cm_y = [659]
#cm = [[214,659]]
#aire = [359*270]
aire1 = 359*270
aire = []
Xc = 33442.0
Yc = -76290.0
Zc = -53546.0
X1 = 36282.703
Y1 = -76981.109
Z1 = -54371.098
d1 = np.sqrt((X1-Xc)**2+(Y1-Yc)**2+(Z1-Zc)**2)
d_list = []
A_list =[]
d = 0
start = 1
string = '/home/kdesousa/Documents/GitHub/TFE/Kalman/2021_04_06_15_40_39_604/cam_00/00_'
data_csv =0
Aire_csv =0 
while(os.path.isfile(string+str(count).zfill(8)+'.csv') == 1  ):
    if start ==1:
        data_1 = pd.read_csv (string+str(count-1).zfill(8)+'.csv',sep=';')
        data_1['XPos'] = data_1['XPos'] - Xc
        data_1['YPos'] = data_1['YPos'] - Yc
        data_1['ZPos'] = data_1['ZPos'] - Zc
        d_1 = np.sqrt(data_1['XPos']**2+data_1['YPos']**2+data_1['ZPos']**2)
        
        start = 0
    else: 
        data_1 = data_csv
        d_1= d
        
    data_csv = pd.read_csv (string+str(count).zfill(8)+'.csv',sep=';')
    data_csv['XPos'] = data_csv['XPos'] - Xc
    data_csv['YPos'] = data_csv['YPos'] - Yc
    data_csv['ZPos'] = data_csv['ZPos'] - Zc
    d = np.sqrt(data_csv['XPos']**2+data_csv['YPos']**2+data_csv['ZPos']**2)/100
    stringcar = data_1['ID']
    stringcsv = data_csv['ID']
    
    
    index = np.zeros(len(stringcar))
    for i in stringcar:
        ok  = data_csv[data_csv['ID'].str.endswith(i)]
        if not len(ok.index) == 0:
            ok_1 = data_1[data_1['ID'].str.endswith(i)]
            inter = (d[ok.index].values - d_1[ok_1.index].values)/d_1[ok_1.index].values
            
            d_list = np.append(d_list,inter)
            Aire = find_ground_truths(ok)
            Aire_1 = find_ground_truths(ok_1)
            delta = (Aire-Aire_1)/Aire_1
            # if np.sign(delta) == - np.sign(inter):
            #     print('oups')
                
                
            if len(delta)>1:
                print('oups1',delta)
            A_list = np.append(A_list,delta)
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    count +=10
plt.scatter(A_list,d_list)
plt.xlim([-0.5,0.5])
plt.ylim([-0.05,0.05])
plt.histogram(A_list,density = True)
plt.show()
plt.histogram(d_list,density = True)
plt.show()
H, xedges, yedges = np.histogram2d(A_list, d)
H = H.T
plt.imshow(H, interpolation='nearest', origin='lower',\
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # X = np.zeros(len(data_csv))
    # X[:] = data_csv[:,3]
    # Y = np.zeros(len(data_csv))
    # Y[:] = data_csv[:,4]
    # Z = np.zeros(len(data_csv))
    # Z[:] = data_csv[:,5]
    # #denormalisation des datas de yolo
    
    # data2[:,1] = data2[:,1]*W#coord x du centre de masse
    # data2[:,2] = data2[:,2]*H#coord y du centre de masse
    # data2[:,3] = data2[:,3]*W#largeur du rectangle
    # data2[:,4] = data2[:,4]*H#hauteur du rectangle

    # cm2 = np.int32(np.array([data2[:,1].tolist(),data2[:,2].tolist()]).T)
    # #value = cm[-1]
    # L1 = np.sqrt(((cm_x[-1]-cm2[:,0])**2+(cm_y[-1]-cm2[:,1])**2)).tolist()
    # index_i1 = L1.index(min(L1))
    # L2 = np.sqrt(((X1-X)**2+(Y1-Y)**2+(Z1-Z)**2)).tolist()
    # index_i2 = L2.index(min(L2))
    # cm_x = np.append(cm_x,cm2[index_i1,0])
    # cm_y = np.append(cm_y,cm2[index_i1,1])
    # prod = data2[index_i1,3]*data2[index_i1,4]
    # aire2 = prod
    # X2 = data_csv[index_i2,3]
    # Y2 = data_csv[index_i2,4]
    # Z2 = data_csv[index_i2,5]
    # d2 = np.sqrt((X2-Xc)**2+(Y2-Yc)**2+(Z2-Zc)**2)
    # #aire = np.append(aire,prod)
    # aire = np.append(aire,(aire2-aire1)/aire1)
    # d = np.append(d,((d2-d1)/d1))
    # aire1 = aire2
    # d1 = d2
    # count+=1
#tt = np.histogram(aire,d,density=True)
#H, xedges, yedges = np.histogram2d(aire,d[d<0.01])
#plt.imshow(H.T, interpolation='nearest', origin='lower',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#plt.plot(d,aire)
#plt.xlabel('Changement de distance')
#plt.ylabel('Changement d aire')
#plt.xlim([-0.01,0.01])
"""a = (set(d))
b = np.array(list(a))
order = b.argsort()
sorted_b = np.take(b, order, 0)
plt.hist(aire,bins=sorted_b)"""


#plt.hist(aire,b)
"""
t = np.zeros(301)
y = np.zeros(301)
for i in range(len(t)):
    t[i] = i
    y[i] = 100000/np.sqrt(t[i])
plt.plot(t,y)
"""