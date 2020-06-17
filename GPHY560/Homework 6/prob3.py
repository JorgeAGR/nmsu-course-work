#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:03:36 2020

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

class Jacobian_LSF(object):
    
    def __init__(self, G, d, f, df):
        self.N = G.shape[1]
        self.G = G
        self.d = d
        self.f = f
        self.df = df
        return
    
    def fit(self, m_0, max_iter=10):
        '''    
        Outputs:
        m : (N+1)-array
            model weights
        m_var : (N+1)*(N+1) matrix
            covariance matrix
        '''
        d = self.d.reshape((len(self.G), 1))
        self.m = m_0
        for i in range(max_iter):
            d_i = d - self.f(self.G, self.m)
            J = self.df(self.G, self.m)
            JTJinv = np.linalg.inv(J.T @ J)
            self.m = self.m + (JTJinv @ J.T) @ d_i
            self.norm = d_i.T @ d_i
        
        self.m = self.m.flatten()
        d_model = self.f(self.G, self.m)
        r = self.d - d_model
        self.data_var = (r**2).sum() / (J.shape[0] - J.shape[1])
        
        self.m_cov = JTJinv * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        return self.m, self.m_cov

# Part A
G = np.array([[1], [1], [1]])
d = np.array([5, 2, 0])

model = LSF(G, d)
xp, cov = model.fit()
x = 1/xp


print('Linearized Fit:')
print('x = {}'.format(x[0]))

# Part B
# Jacobian method
f = lambda G, m: G / m
df = lambda G, m: -G/m**2

model_b = Jacobian_LSF(G, d, f, df)
x_b, cov_b = model_b.fit(1/2)

print('Jacobian Fit:')
print('x = {}'.format(x_b[0]))