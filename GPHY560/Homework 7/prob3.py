#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:54:37 2020

@author: jorgeagr
"""
import numpy as np 

# Damped LSF
class Damped_LSF(object):
    
    def __init__(self, G, d, eps):
        self.N = G.shape[1]
        self.G = G
        self.dof = self.G.shape[0] - self.N
        self.d = d
        self.eps = eps
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
        GTGinv = np.linalg.inv(G.T @ G + np.eye(self.N)*self.eps)
        
        self.m = (GTGinv @ G.T) @ self.d
        
        d_model = self.G @ self.m
        r = self.d - d_model
        self.sse = (r**2).sum()
        self.data_var = self.sse / self.dof
        
        self.m_cov = GTGinv @ G.T @ G @ GTGinv * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        return self.m, self.m_cov

    # Calculate output for the model given input data
    def func(self, g):
        d = np.dot(g, self.m).flatten()
        d_var = g @ (self.m_cov @ g.T)
        return d, d_var
    
G = np.array([[1, 1],
              [1,0],
              [0,1]])
d = np.array([np.log(10), 10, 10])

# Damping factor
eps = 1

model = Damped_LSF(G, d, eps)
m, cov = model.fit()

print('Model parameters:')
print('x = {}'.format(m[0]))
print('y = {}'.format(m[1]))