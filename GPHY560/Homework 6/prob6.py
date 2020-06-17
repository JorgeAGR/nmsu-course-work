#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 01:03:59 2020

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
            d_i = d - self.f(self.G, self.m).reshape((len(self.G), 1))
            J = self.df(self.G, self.m)
            JTJinv = np.linalg.inv(J.T @ J)
            self.m = self.m + (JTJinv @ J.T) @ d_i
            self.norm = d_i.T @ d_i
        
        self.m = self.m.flatten()
        d_model = self.f(self.G, self.m)
        r = d - d_model
        self.data_var = (r**2).sum() / (J.shape[0] - J.shape[1])
        
        self.m_cov = JTJinv @ J.T @ J @ JTJinv * self.data_var
        self.m_std = np.sqrt(self.m_cov.diagonal())
        
        return self.m, self.m_cov

v = 6

G = np.array([[0, 0],
     [10, 10],
     [10, 0],
     [0, 10]])

d = np.array([3, 2.12, 2.35, 2.53])

def f(G, m):
    '''
    m[0] == x_0
    m[1] == y_0
    m[2] == t_0
    '''
    return m[2] + np.sqrt((G[:,0]-m[0])**2 + (G[:,1]-m[1])**2)/v

def df(G, m, h=0.001):
    # Centered differencing
    J = np.ones((len(G), len(m)))
    for i in range(2):
        J[:,i] = -(G[:,i] - m[i]) / (v * np.sqrt((G[:,0]-m[0])**2 + (G[:,1]-m[1])**2))
    return J

model = Jacobian_LSF(G, d, f, df)
m, cov = model.fit(np.array([[5], [5], [0]]))

print('Solution:')
print('x_0 = {:.2f} km'.format(m[0]))
print('y_0 = {:.2f} km'.format(m[1]))
print('t_0 = {:.2f} s'.format(m[2]))