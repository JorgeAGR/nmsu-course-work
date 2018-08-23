#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 20:29:32 2018

@author: jorgeagr
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [width, height]


def rotate(angle, axis):
    T = np.zeros(shape = (3,3))
    angle = np.deg2rad(angle)
    if axis == 'x':
        T[0][0] = 1
        T[1][1] = T[2][2] = np.cos(angle)
        T[2][1] = -np.sin(angle)
        T[1][2] = np.sin(angle)
        return T
    if axis == 'y':
        T[1][1] = 1
        T[0][0] = T[2][2] = np.cos(angle)
        T[0][2] = -np.sin(angle)
        T[2][0] = np.sin(angle)
        return T
    if axis == 'z':
        T[2][2] = 1
        T[0][0] = T[1][1] = np.cos(angle)
        T[1][0] = -np.sin(angle)
        T[0][1] = np.sin(angle)
        return T

T0 = rotate(0, 'x')
Ta = rotate(45, 'x')
Tb = rotate(90, 'y')
Tc = rotate(90, 'y')
Td = rotate(45, 'x')

T = [T0, Ta, Tb, Tc, Td]
A = np.array([[1], [2], [3]])
origin = 0, 0, 0

text = 'Original', 'Rotation A', 'Rotation B', 'Rotation C', 'Rotation D'

for i in range(len(T)):
    A = np.dot(T[i], A)
    fig = plt.figure()
    fig.suptitle(text[i])
    print(A)
    
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_xlim([-5, 5])
    ax.set_xlabel('x')
    ax.set_ylim([-5, 5])
    ax.set_ylabel('y')
    ax.set_zlim([-5, 5])
    ax.set_zlabel('z')
    ax.quiver(origin[0], origin[1], origin[2], A.flatten()[0], A.flatten()[1], A.flatten()[2])