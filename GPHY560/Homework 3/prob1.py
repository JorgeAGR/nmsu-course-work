#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:04:56 2020

@author: jorgeagr
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Setting plotting variables
width = 10
height = 10

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 18
mpl.rcParams['figure.titlesize'] = 'small'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 18

class Constrained_LSF(object):
    
    def __init__(self, G, d, F, h):
        self.N = G.shape[1]
        self.G = G
        self.d = d.reshape(len(d),1)
        self.F = F
        self.h = h.reshape(len(h),1)
        return
    
    # LSF function to calcualte model parameters and covariance matrix
    def fit(self):
        '''    
        Outputs:
        m : (N+1)-array
            model weights
        m_var : (N+1)*(N+1) matrix
            covariance matrix
        '''
        G = self.G
        GT = self.G.T
        F = self.F
        FT = self.F.T
        GTGinv = np.linalg.inv(GT @ G)
        
        self.m = (GTGinv @ GT) @ self.d
        # Recalculate m for constrained version
        self.m = self.m - GTGinv @ FT @ np.linalg.inv(F @ GTGinv @ FT) @ (F @ self.m - h)
        
        d_model = G @ self.m
        r = self.d - d_model
        self.data_var = (r**2).sum() / (G.shape[0] - self.N + len(self.h))
        
        I = np.eye(len(self.m))
        self.m_cov = (I - GTGinv @ FT @ np.linalg.inv(F @ GTGinv @ FT) @ F) @ GTGinv @ (I - GTGinv @ FT @ np.linalg.inv(F @ GTGinv @ FT) @ F) * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        return self.m, self.m_cov

    # Calculate output for the model given input data
    def func(self, g):
        d = np.dot(g, self.m).flatten()
        d_var = g @ (self.m_cov @ g.T)
        return d, d_var

def minsec2dec(degrees, minutes, seconds):
    degrees += (minutes/60) + (seconds/3600)
    return degrees

def dec2minsec(degrees):
    remainder = degrees
    degrees = int(degrees)
    remainder = remainder - degrees
    minutes = int(remainder*60)
    remainder = remainder*60 - minutes
    seconds = int(round(remainder*60))
    return degrees, minutes, seconds

A = minsec2dec(40, 19, 2)
B = minsec2dec(70, 30, 1)
C = minsec2dec(69, 11, 5)

d = np.vstack([A, B, C])
G = np.eye(3)

F = np.array([[1, 1, 1]])
h = np.array([180, ])

model = Constrained_LSF(G, d, F, h)
m, cov = model.fit()

# Part A
print('Approximate Angles')
for l, a in zip(('A', 'B', 'C'), (m.flatten())):
    deg = dec2minsec(a)
    print('{} = {} deg {} min {} sec'.format(l, deg[0], deg[1], deg[2]))
print('\nSum of angles = {}'.format(m.sum()))
# Part B
print('\nData Variance:', model.data_var)

# Part C
print('\nCovariance Matrix:\n', model.m_cov)