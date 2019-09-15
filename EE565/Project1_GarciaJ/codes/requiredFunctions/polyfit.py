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
    
    def fit_LS(self, train_data, y_data, reg=0):
        
        phi = np.ones((train_data.shape[0], self.m_powers))
        
        for m in range(self.m_powers):
            phi[:, m] = train_data ** m
        
        y_data = np.reshape(y_data, (y_data.shape[0], 1))
        phi_H = phi.transpose().conjugate()
        phi_dag = np.dot(np.linalg.inv((np.identity(self.m_powers) * reg + np.dot(phi_H, phi))), 
                     phi_H)
        self.weights = np.dot(phi_dag, y_data)
        
        return
        
    def predict(self, x_data):
        
        phi = np.ones((x_data.shape[0], self.m_powers))
        
        for m in range(self.m_powers):
            phi[:, m] = x_data ** m
        
        return np.dot(phi, self.weights)