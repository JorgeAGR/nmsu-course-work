#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:25:05 2019

@author: jorgeagr
"""

import numpy as np

class KNN(object):
    
    def __init__(self, train_x, train_class):
        
        self.train_x = train_x
        self.train_class = train_class
        
        return
    
    def predict(self, x_data, k_neighbors):
        
        distancesq = (((x_data - self.train_x)**2).sum(axis=0))
        nn_ind = np.argsort(distancesq)[:k_neighbors]
        nn_class, counts = np.unique(self.train_class[nn_ind], return_counts=True)
        
        return nn_class[np.argmax(counts)]