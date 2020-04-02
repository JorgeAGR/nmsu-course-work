#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:02:34 2020

@author: jorgeagr
"""

import numpy as np

# Normal LSF
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

# LSF for underdetermined data
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
        
        self.m_cov = G.T @ GGTinv @ GGTinv @ G * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        return self.m, self.m_cov

    # Calculate output for the model given input data
    def func(self, g):
        d = np.dot(g, self.m).flatten()
        d_var = g @ (self.m_cov @ g.T)
        return d, d_var

G = np.array([[1, 1, 0],
              [0, 0, 1],
              [0, 0, 1]])
    
d = np.array([[1], [2], [1]])

# Part C
G_c = G[1:,-1].reshape(2,1)
d_c = d[1:]

model_c = LSF(G_c, d_c)
m_c, cov_c = model_c.fit()
print('Overdetermined parameter: m3 = {}'.format(m_c[0][0]))

# Part D
G_d = G[0, :2].reshape((1,2))
d_d = d[0]
model_d = LSF_UD(G_d, d_d)
m_d, cov_d = model_d.fit()
print('Underdetermined parameters: m1 = {}, m2 = {}'.format(m_d[0], m_d[1]))

# Part E
m = np.hstack((m_d, m_c[0])).reshape(3,1)
# Need to add a small dampening factor to invert d * d^T since its determinant is 0:
Gg = (m @ d.T @ np.linalg.inv(d@d.T + np.eye(3) * 0.001))
print('Generalized inverse: \nG^g = {}'.format(Gg))

# Part F
var = (G@m - d).T @ (G@m - d)
print('Data Variance: {}'.format(var[0][0]))

# Part G
# cov(m) = G^-g cov(d) G^-gT = G^-g G^-gT var (from Menke Ch. 7 eq. 7.43)
cov = Gg @ Gg.T * var
print('Parameter Covariance: \ncov(m) = {}'.format(cov))

# Part H
# N = G G^-g
N = G @ Gg
print('Data Resolution Matrix: \nN = {}'.format(N))

# Part I
# R = G^-g G
R = Gg @ G
print('Model Resolution Matrix: \nR = {}'.format(R))