# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:53:16 2021

@author: Gauthier_Rotsart
"""

import cv2
import numpy as np
import glob
c = 0
img_array = []
for filename in glob.glob('C:/Users/Gauthier_Rotsart/Documents/yolov5/runs/detect/exp4/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()