#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:24:47 2019

@author: jorgeagr
"""

import numpy as np

class kMeans(object):
    
    def __init__(self):
        return
    
    def fit(self, X, k, online=False, centroids=None, epsilon=1e-4, max_iter=100):
        if online:
            self.fit_batch(X, k, centroids, epsilon, max_iter)
            return
        else:
            return
    
    def fit_batch(self, X, k, centroids, epsilon, max_iter):
        old_centroids = np.zeros(k, X.shape[1])
        if not centroids:
            centroids = np.random.rand(k, X.shape[1])
        avg_change = np.abs((centroids - old_centroids)).mean()
        
        iters = 0
        while (avg_change > epsilon) or (iters < max_iters):
            
            
            
        return