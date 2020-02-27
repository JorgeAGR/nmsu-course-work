#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:27:55 2020

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
    def __init__(self, G, y):
        self.N = G.shape[1]
        self.G = G
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
        self.data_var = (r**2).sum() / (self.G.shape[0] - self.N)
        self.m_cov = np.linalg.inv(np.dot(self.G.T, self.G)) * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        return self.m, self.m_cov

    # Calculate output for the model given input data
    def func(self, g):
        d = np.dot(g, self.m).flatten()
        d_var = np.dot(g, np.dot(self.m_cov, g.T))
        return d, d_var
    
x1 = np.array([0, 1, 1])
x2 = np.array([1, 0, 1])
y1 = np.array([-2.1, 1, -0.9])
y2 = np.array([1.1, 2, 2.8])

G = np.ones((6, 2))
G[:3,0] =  x1
G[3:,0] = x2
G[:3,1] = -x2
G[3:,1] = x1
d = np.hstack([y1, y2])

model = LSF(G, d)
model.fit()
a = model.m[0][0]
b = model.m[1][0]

# Part A
print('Model Parameters')
print('a =', a)
print('b =', b)

# Part B
print('\nData Variance:', model.data_var)

# Part C
print('\nCovariance Matrix:\n', model.m_cov)

# Part D
x1_new = 1
x2_new = 2
g = np.ones((2,2))
g[0,0] = x1_new
g[0,1] = -x2_new
g[1,0] = x2_new
g[1,1]= x1_new

y_new, y_new_var = model.func(g)

print('New point:', '(' + str(y_new[0]) + ',' + str(y_new[1]) + ')')

# Part E
print('New Point Covariance:\n', y_new_var)