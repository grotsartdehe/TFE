#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:16:45 2021

@author: kdesousa
"""
import os 
#create tracks
folder = '/home/kdesousa/Documents/GitHub/TFE/Kalman/Data/radar/data-yolo-2'
for i in range(288):
    name = 'data_'+ str(i).zfill(4)+ '.txt'
    name = os.path.join(folder,name)
    file = open(name,'w')
    file.write('d v \n') 
    if i > 119  and i < 204 :
        lol = str(i/40) + ' ' + str(40)
        file .write(lol + '\n')
    elif i>= 205 and i < 287 : 
        lol = str(i/40) + ' ' + str(40)
        file.write(lol + '\n')
        lol = str(i/80) + ' ' + str(3)
        file.write(lol + '\n')
