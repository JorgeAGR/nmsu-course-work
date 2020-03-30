#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 00:27:26 2020

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

class Weighted_Constrained_LSF(object):
    
    def __init__(self, G, d, F, h, error, delta):
        self.N = G.shape[1]
        self.G = np.vstack([G, F])
        self.d = np.vstack([d.reshape(len(d),1), h.reshape(len(h),1)])
        self.W = np.eye(len(error) + len(delta)) * np.concatenate((error, delta))
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
        G = self.W @ self.G
        GTGinv = np.linalg.inv(G.T @ G)
        d = self.W @ self.d
        
        self.m = (GTGinv @ G.T) @ d
        
        d_model = self.G @ self.m
        r = self.d - d_model
        self.data_var = (r**2).sum() / (G.shape[0] - self.N)
        
        self.m_cov = GTGinv * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        return self.m, self.m_cov

    # Calculate output for the model given input data
    def func(self, g):
        d = np.dot(g, self.m).flatten()
        d_var = g @ (self.m_cov @ g.T)
        return d, d_var
    
class Weighted_LSF(object):
    
    def __init__(self, G, d, w):
        self.N = G.shape[1]
        self.G = G
        self.d = d.reshape(len(d),1)
        self.W = np.eye(len(w)) * w
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
        G = self.W @ self.G
        GTGinv = np.linalg.inv(G.T @ G)
        d = self.W @ self.d
        
        self.m = (GTGinv @ G.T) @ d
        
        d_model = self.G @ self.m
        r = self.d - d_model
        self.data_var = (r**2).sum() / (G.shape[0] - self.N)
        
        self.m_cov = np.linalg.inv(G.T @ G) * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        return self.m, self.m_cov

    # Calculate output for the model given input data
    def func(self, g):
        d = np.dot(g, self.m).flatten()
        d_var = g @ (self.m_cov @ g.T)
        return d, d_var

elevation = np.array([3.1, 0.4, -3.3])
distance = np.array([500, 1000, 750])
error = 1/distance
G = np.eye(3)   

# Part A
F = np.array([[1, 1, 1]])
h = np.array([0, ])
delta = np.array([100, ])

model = Weighted_Constrained_LSF(G, elevation, F, h, error, delta)
m, m_cov = model.fit()

print('Weighted Constrained LSF')
print('---')
print('Estimated Elevations:')
for l, h in zip(('A-B', 'B-C', 'C-A'), (m.flatten())):
    print('{} = {} m'.format(l, h))
print('Sum of elevations = {}'.format(m.sum()))
print('Covariance:')
print(m_cov)
G = np.array([[1, 0],
              [0, 1],
              [-1, -1]])
model2 = Weighted_LSF(G, elevation, error)
m2, m2_cov = model2.fit()
print('\nSubstitution + LSF')
print('---')
print('Estimated Elevations:')
for l, h in zip(('A-B', 'B-C'), (m2.flatten())):
    print('{} = {} m'.format(l, h))
print('C-A = {}'.format(-m2.sum()))
print('Covariance:')
print(m2_cov)

