# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 22:42:25 2020

@author: Kdesousa
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')


fig = plt.figure()
ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,
def animate(i):
    x1 = np.linspace(0, 4, 1000)
    y1= np.ones(1000)*4
    y2 = np.linspace(4, 0, 1000)
    x2= np.ones(1000)*4
    x3 = np.linspace(4, 0, 1000)
    y3 = np.ones(1000)*0
    y4 = np.linspace(0, 4, 1000)
    x4 = np.ones(1000)*0

    x = np.append(x1,x2)
    xx = np.append(x3,x4)
    x = np.append(x,xx)-i/25
    y = np.append(y1,y2)
    yy = np.append(y3,y4)
    y= np.append(y,yy)
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=200, blit=True)
anim.save('sine_wave.gif', writer='imagemagick')