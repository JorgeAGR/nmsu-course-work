#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:02:45 2020

@author: jorgeagr
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Using damped LSF as suggested by Menke (Ch 12)
class LSF(object):
    
    def __init__(self, G, d):
        self.N = G.shape[1]
        self.G = G
        self.d = d
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
        GTGinv = np.linalg.inv(G.T @ G)
        
        self.m = (GTGinv @ G.T) @ self.d
        
        d_model = self.G @ self.m
        r = self.d - d_model
        self.data_var = (r**2).sum() / (G.shape[0] - self.N)
        
        self.m_cov = GTGinv @ G.T @ G @ GTGinv * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        return self.m, self.m_cov

    # Calculate output for the model given input data
    def func(self, g):
        d = np.dot(g, self.m).flatten()
        d_var = g @ (self.m_cov @ g.T)
        return d, d_var

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

f_points = 100
filter_points = 7

# Build function
f = np.zeros(f_points)
f[0:7] = 1/7

# Build G of function
G = np.zeros((f_points, filter_points))
for i in range(filter_points):
    G[i:i+f_points, i] = f[:f_points-i]
    
# Delta function
d = np.zeros(f_points)
d[0] = 1

# m is the inverse filter
model = LSF(G, d)
m, cov = model.fit()

fig = plt.figure()
ax = [plt.subplot2grid((4,4), (0,0), colspan=2, rowspan=2, fig=fig),
      plt.subplot2grid((4,4), (0,2), colspan=2, rowspan=2, fig=fig),
      plt.subplot2grid((4,4), (2,1), colspan=2, rowspan=2, fig=fig)]
ax[0].plot(f)
ax[1].plot(m)
ax[2].plot(np.convolve(f, m))
ax[2].plot(d)