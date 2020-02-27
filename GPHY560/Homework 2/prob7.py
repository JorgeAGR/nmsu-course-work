#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:27:58 2020

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


class LSF(object):
    '''
        Inputs:
        x : array
            input data
        y : array
            output data
        N : int
            order of fit
    '''
    def __init__(self, x, y, N):
        self.N = N+1
        self.G = np.ones((len(x), self.N))
        for i in np.arange(self.N):
            self.G[:,i] = x**i
        self.d = y.reshape(len(y),1)
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
        self.m = np.dot(np.dot(np.linalg.inv(np.dot(self.G.T, self.G)), self.G.T), self.d)
        
        d_model = np.dot(self.G, self.m)
        r = self.d - d_model
        self.data_var = (r**2).sum() / (len(x) - self.N)
        self.m_cov = np.linalg.inv(np.dot(self.G.T, self.G)) * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        return self.m, self.m_cov

    # Calculate output for the model given input data
    def func(self, x):
        g = np.ones((len(x), self.N))
        for i in np.arange(self.N):
            g[:,i] = x**i
        d = np.dot(g, self.m).flatten()
        d_var = np.dot(g, np.dot(self.m_cov, g.T)).flatten()
        return d, d_var

t = np.array([0, 1, 2, 5, 7, 11])
x = np.array([0, 1, 2, 3, 4, 5])

model = LSF(t, x, 2)
m, m_cov = model.fit()

# Part A
print('Fit Results')
print('Acceleration:', 2*m[2])
print('Velocity:', m[1])
print('Initial Position', m[0])

# Part B
print('\nData Variance:', model.data_var)

# Part C
print('\nAcceleration Std Dev:', 2*model.m_std[2])
print('Velocity Std Dev:', model.m_std[1])
print('Init Position Std Dev:', model.m_std[0])

# Part D
x_new, x_new_var = model.func(np.array([8]))
print('\nx(8) =', x_new)

# Part E
print('V(x(8)) =', x_new_var)