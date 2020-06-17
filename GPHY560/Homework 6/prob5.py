#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:55:13 2020

@author: jorgeagr
"""

import numpy as np

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
        
        d_model = self.f(self.G, self.m)
        self.m = self.m.flatten()
        r = d - d_model
        self.data_var = (r**2).sum() / (J.shape[0] - J.shape[1])
        
        self.m_cov = JTJinv * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        return self.m, self.m_cov

G = np.array([[-6], [-2], [2]])
d = np.array([1.6, 3, 3])

# m[0] == s^2
# m[1] == A
f = lambda G, m: m[1] * np.exp(-G**2/(2*m[0]))
def df(G, m):
    J = np.zeros((len(G), len(m)))
    J[:,0] = (- G / m[0] * m[1] * np.exp(-G**2/(2*m[0]))).flatten()
    J[:,1] = (np.exp(-G**2/(2*m[0]))).flatten()
    return J

model = Jacobian_LSF(G, d, f, df)
m, cov = model.fit(np.array([[15], [1.5]]), 200)

print('Jacobian Fit')
print('s^2 = {:.4f}'.format(m[0]))
print('A = {:.4f}'.format(m[1]))