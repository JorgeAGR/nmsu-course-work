#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 07:08:12 2018

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


V = 5
L = 8

k = np.linspace(0, 4*np.pi, num = 800)

def K1(x):
    return np.sqrt((2 * V) - k**2)

def K2(x):
    return - k * 1/np.tan(k*L/2)

tan = K2(k)
limit = 5
tan[tan > limit*3] = np.inf
tan[tan < -limit*3] = -np.inf

plt.plot(k, tan)
plt.plot(k, K1(k))
plt.ylim(0,5)
plt.xlim(0, 4)
plt.xlabel('k')
plt.ylabel(r'$\kappa$')
plt.tight_layout()

'''
V = 5
a = 10

z0 = a * np.sqrt(2 * V)

z = np.linspace(0, 4*np.pi, num = 600)

limit = 30

tanz = -1/np.tan(z)
tanz[tanz > limit*3] = np.inf
tanz[tanz < -limit*3] = -np.inf

z2 = np.sqrt((z0/z)**2 - 1)

fig, ax = plt.subplots()
ax.plot(z, tanz)
ax.plot(z, z2)
ax.set_ylim(0,limit*0.75)
ax.set_xlim(0,4*np.pi)
'''