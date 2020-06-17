#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:18:14 2020

@author: jorgeagr
"""

import numpy as np

# Vanilla LSF
class LSF(object):
    
    def __init__(self, G, d):
        self.G = G
        self.dof = G.shape[0] - G.shape[1]
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
        self.sse = (r**2).sum()
        self.data_var = self.sse / self.dof
        
        self.m_cov = GTGinv @ G.T @ G @ GTGinv * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        # Data resolution matrix
        self.N = G@GTGinv@G.T
        
        return self.m, self.m_cov
    
G = np.ones(5).reshape((5,1))
d = np.array([1, 2, 3, 4, 5])

model = LSF(G, d)
m, cov = model.fit()
# Part A
print('Model Parameters: m = {}'.format(m[0]))

# Part B
print('Data variance: {}'.format(model.data_var))
print('Model variance: var(m) = {}'.format(cov.flatten()[0]))

# Part C
print('Data resolution matrix:\n{}'.format(model.N))