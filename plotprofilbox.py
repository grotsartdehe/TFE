#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:16:35 2021

@author: kdesousa
"""
import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.optimize as optimization
def hist_eq(img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function will do histogram equalization on a given 1D np.array
    meaning will balance the colors in the image.
    For more details:
    https://en.wikipedia.org/wiki/Histogram_equalization
    **Original function was taken from open.cv**
    :param img: a 1D np.array that represent the image
    :return: imgnew -> image after equalization, hist-> original histogram, histnew -> new histogram
    trouvé sur https://stackoverflow.com/questions/61178379/how-to-do-histogram-equalization-without-using-cv2-equalizehist
    """

    # Flattning the image and converting it into a histogram
    histOrig, bins = np.histogram(img.flatten(), 256, [0, 255])
    # Calculating the cumsum of the histogram
    cdf = histOrig.cumsum()
    # Places where cdf = 0 is ignored and the rest is stored
    # in cdf_m
    cdf_m = np.ma.masked_equal(cdf, 0)
    # Normalizing the cdf
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # Filling it back with zeros
    cdf = np.ma.filled(cdf_m, 0)


    # Creating the new image based on the new cdf
    imgEq = cdf[img.astype('uint8')]
    histEq, bins2 = np.histogram(imgEq.flatten(), 256, [0, 256])

    return imgEq, histOrig, histEq

    pass
A_list  = np.array((np.load('A.npy')))
d_list =np.array(( np.load('d.npy')))
cond2 = d_list>-0.4  
cond1 = A_list <2
cond = cond1 & cond2

#plt.scatter(A_list[cond],d_list[cond])
# def func(x, a, b,c):
#     return a + b*x+c*x*x 
# x0 = np.zeros(3)
# l = optimization.curve_fit(func, A_list ,d_list, x0)
# print(l)
# #A =  np.histogram2d(A_list[cond],d_list[cond])

# plt.scatter(A_list,d_list)
# plt.scatter(A_list,l[0][0]+l[0][1]*A_list+l[0][2]*A_list*A_list)
# plt.xlabel('Delta A')
# plt.ylabel('Delta d')
# plt.title("Evolution de la diff de distance en fonction de la diff d'aire ")
plt.figure()

# #plt.xlim([-0.5,0.5])
plt.ylim([-0.2,0.2])

H, xedges,yedges = np.histogram2d(A_list,d_list,density=True)
X, Y = np.meshgrid(xedges, yedges)
plt.pcolormesh(X, Y, H.T)
plt.colorbar()

fig = plt.figure()
im = plt.NonUniformImage(fig, interpolation='bilinear')
xcenters = (xedges[:-1] + xedges[1:]) / 2
ycenters = (yedges[:-1] + yedges[1:]) / 2
im.set_data(xcenters, ycenters, H)
fig.images.append(im)
plt.show()

