#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:09:32 2020

@author: jorgeagr
"""

import numpy as np

# Vanilla LSF
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

G = np.array([[-6, 1], [-2, 1], [2, 1]])
d = np.array([1.6, 3, 3])

G = G**2
ln_d = np.log(d)

model = LSF(G, ln_d)
m, cov = model.fit()

# m[0] => s^2
# m[1] => A
# Slope
s2p = -2*m[0]
s2 = 1/s2p

# Intercept
Ap = m[1]
A = np.exp(Ap)

print('Linearized Fit')
print('s^2 = {:.4f}'.format(s2))
print('A = {:.4f}'.format(A))