#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:18:03 2020

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

class Weighted_LSF(object):
    
    def __init__(self, G, d, var):
        self.N = G.shape[1]
        self.G = G
        self.d = d.reshape(len(d),1)
        self.W = np.eye(len(var)) * (1 / np.sqrt(var))
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

data = np.array([1, 2, 4, 8]).reshape(4,1)
variance = np.array([1, 2, 3, 4])
G = np.ones(len(variance)).reshape(4,1)

model = Weighted_LSF(G, data, variance)
m, cov = model.fit()

print('Estimate of m:')
print('m = {}'.format(m.flatten()[0]))