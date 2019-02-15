#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 00:57:33 2018

@author: jorgeagr
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

g = 32.2

# 2
def x_2(t):
    return (16 * np.sqrt(2*g) / (g-32)) * np.sin(np.sqrt(g/2) * t) + (4 * g / (g-32)) * np.sin(4*t)

# 3
def x_3(t):
    return (30 * np.sqrt(2*g) / (g-72)) * np.sin(np.sqrt(g/2) * t) + (5 * g / (g-72)) * np.cos(6*t)

t = np.linspace(0, 10, num=500)

fig, ax = plt.subplots()
ax.plot(t, x_2(t))
ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [ft]')
ax.set_title('Homework 4 - Problem 2', weight = 'bold')
ax.grid()
ax.set_xlim(0, max(t))
plt.tight_layout()

fig, ax = plt.subplots()
ax.plot(t, x_3(t))
ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [ft]')
ax.set_title('Homework 4 - Problem 3', weight = 'bold')
ax.grid()
ax.set_xlim(0, max(t))
plt.tight_layout()