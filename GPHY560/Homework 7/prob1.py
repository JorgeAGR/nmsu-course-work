#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:41:25 2020

@author: jorgeagr
"""
import numpy as np
from scipy.stats import f as f_dist

# Vanilla LSF
class LSF(object):
    
    def __init__(self, G, d):
        self.N = G.shape[1]
        self.G = G
        self.dof = self.G.shape[0] - self.N
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
        
        return self.m, self.m_cov

    # Calculate output for the model given input data
    def func(self, g):
        d = np.dot(g, self.m).flatten()
        d_var = g @ (self.m_cov @ g.T)
        return d, d_var
    

def fourier(x, n):
    G = np.zeros((x.shape[0], 2*n))
    func = {0: np.sin, 1: np.cos}
    for i in range(2*n):
        G[:,i] = func[i%2]((i//2+1)*x)
    return G

x = np.array([0.5, 1, 1.5, 2, 2.5, 3.5, 4, 4.5, 5, 5.5])
y = np.array([3.02, -0.2, -2.08, -1, 0.92, -0.58, -3.47, -4.56, -2.29, 1.6])

zero = False
n = 0
while zero == False:
    n += 1
    G1 = fourier(x, n)
    model1 = LSF(G1, y)
    m1, cov1 = model1.fit()
    
    m = n + 1
    G2 = fourier(x, m)
    model2 = LSF(G2, y)
    m2, cov2 = model2.fit()
    
    # Hypothesis: Model1 = Model2
    f1 = (model1.sse - model2.sse) / (model1.dof - model2.dof)
    f2 = model2.sse / model2.dof
    F = f1/f2
    p_val = f_dist.sf(F, model1.dof - model2.dof, model2.dof)
    if p_val < 0.05:
        zero = True
        print('Model difference between n = {} and n = {} not significant'.format(n, m))
        print('p-val = {}'.format(p_val))