#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:22:11 2020

@author: jorgeagr
"""

import numpy as np

# LSF for underdetermined data, minimizing solution norm
class LSF_UD(object):
    
    def __init__(self, G, d):
        self.N = G.shape[1]
        self.G = G
        self.d = d
        return
    
    # LSF function to calcualte model parameters and covariance matrix
    def fit(self):
        '''    
        Outputs:Normal LSF
        m : (N+1)-array
            model weights
        m_var : (N+1)*(N+1) matrix
            covariance matrix
        '''
        G = self.G
        GGTinv = np.linalg.inv(G @ G.T)
        
        self.m = (G.T @ GGTinv) @ self.d
        
        d_model = self.G @ self.m
        r = self.d - d_model
        
        self.data_var = (r**2).sum() / (G.shape[0] - self.N)
        
        self.m_cov = G.T @ GGTinv @ GGTinv @ G
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        # Model resolution matrix
        self.R = G.T@GGTinv@G0.
        
        return self.m, self.m_cov

    # Calculate output for the model given input data
    def func(self, g):
        d = np.dot(g, self.m).flatten()
        d_var = g @ (self.m_cov @ g.T)
        return d, d_var

G = np.array([[1, 1, 0, 0],
              [0, 0, 1, 1]])
d = np.array([2, 4])

model = LSF_UD(G, d)
m, cov = model.fit()

# Part A
print('Model Parameters: m = {}'.format(m.flatten()))

# Part B
print('Model variance: var(m) = {}'.format(cov.diagonal()))

# Part C
print('Model resolution matrix:\n{}'.format(model.R))