#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:03:28 2019

@author: jorgeagr
"""

import numpy as np

class PolyFit(object):
    
    def __init__(self, m_powers):
        self.m_powers = m_powers + 1
    
    def fit_LS(self, train_x, train_y, reg=0):
        
        phi = self._build_Data(train_x)
        train_y = np.reshape(train_y, (train_y.shape[0], 1))
        phi_H = phi.transpose().conjugate()
        phi_dag = np.dot(np.linalg.inv((np.identity(self.m_powers) * reg + np.dot(phi_H, phi))), 
                     phi_H)
        self.weights = np.dot(phi_dag, train_y)
        
        return self.weights, self._eval_Error(train_x, train_y)
        
    def test(self, test_x, test_y):
        
        return self.predict(test_x), self._eval_Error(test_x, test_y)
    
    def predict(self, x_data):
        
        phi = self._build_Data(x_data)
        
        return np.dot(phi, self.weights)
    
    def _build_Data(self, x):
        
        phi = np.ones((x.shape[0], self.m_powers))
        
        for m in range(self.m_powers):
            phi[:, m] = x ** m
        
        return phi
    
    def _eval_Error(self, x_data, y_data):
        
        return (1/2)*((self.predict(x_data) - y_data)**2).sum()