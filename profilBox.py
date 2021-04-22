# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 12:49:14 2021

@author: Gauthier_Rotsart
"""
import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt

count = 2261#2000#test de la frame 2437
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
d = []
while(os.path.isfile('C:/Users/Gauthier_Rotsart/Downloads/data-CC/00_'+str(count).zfill(8)+'.txt') == 1):
    data2 = pd.read_csv ('C:/Users/Gauthier_Rotsart/Downloads/data-CC/00_'+str(count).zfill(8)+'.txt',sep=' ',header = None).values
    data_csv = pd.read_csv ('C:/Users/Gauthier_Rotsart/Downloads/data-CC/00_'+str(count).zfill(8)+'.csv',sep=';').values
    X = np.zeros(len(data_csv))
    X[:] = data_csv[:,3]
    Y = np.zeros(len(data_csv))
    Y[:] = data_csv[:,4]
    Z = np.zeros(len(data_csv))
    Z[:] = data_csv[:,5]
    #denormalisation des datas de yolo
    
    data2[:,1] = data2[:,1]*W#coord x du centre de masse
    data2[:,2] = data2[:,2]*H#coord y du centre de masse
    data2[:,3] = data2[:,3]*W#largeur du rectangle
    data2[:,4] = data2[:,4]*H#hauteur du rectangle

    cm2 = np.int32(np.array([data2[:,1].tolist(),data2[:,2].tolist()]).T)
    #value = cm[-1]
    L1 = np.sqrt(((cm_x[-1]-cm2[:,0])**2+(cm_y[-1]-cm2[:,1])**2)).tolist()
    index_i1 = L1.index(min(L1))
    L2 = np.sqrt(((X1-X)**2+(Y1-Y)**2+(Z1-Z)**2)).tolist()
    index_i2 = L2.index(min(L2))
    cm_x = np.append(cm_x,cm2[index_i1,0])
    cm_y = np.append(cm_y,cm2[index_i1,1])
    prod = data2[index_i1,3]*data2[index_i1,4]
    aire2 = prod
    X2 = data_csv[index_i2,3]
    Y2 = data_csv[index_i2,4]
    Z2 = data_csv[index_i2,5]
    d2 = np.sqrt((X2-Xc)**2+(Y2-Yc)**2+(Z2-Zc)**2)
    #aire = np.append(aire,prod)
    aire = np.append(aire,(aire2-aire1)/aire1)
    d = np.append(d,((d2-d1)/d1))
    aire1 = aire2
    d1 = d2
    count+=1
#tt = np.histogram(aire,d,density=True)
#H, xedges, yedges = np.histogram2d(aire,d[d<0.01])
#plt.imshow(H.T, interpolation='nearest', origin='lower',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.plot(d,aire)
#plt.xlabel('Changement de distance')
#plt.ylabel('Changement d aire')
#plt.xlim([-10,10])
a = (set(d))
b = np.array(list(a))
order = b.argsort()
sorted_b = np.take(b, order, 0)
#plt.hist(aire,bins=sorted_b)


#plt.hist(aire,b)
"""
t = np.zeros(301)
y = np.zeros(301)
for i in range(len(t)):
    t[i] = i
    y[i] = 100000/np.sqrt(t[i])
plt.plot(t,y)
"""