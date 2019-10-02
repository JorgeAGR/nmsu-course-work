# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:38:04 2019

@author: jorge
"""

import numpy as np

class FishersDiscriminant(object):
    
    def __init__(self, train_x, train_y):
        mu0, mu1 = self._calc_Means(train_x, train_y)
        self._calc_Weights(mu0, mu1)
        return
    
    def _calc_Weights(self, mu0, mu1):
        
        w = mu1 - mu0
        w_mag = np.sqrt((w**2).sum())
        self.weights = w / w_mag
        return self.weights
    
    def _calc_Means(self, x, y):
        classes = np.unique(y)
        means = np.zeros((len(classes), x.shape[1]))
        for i, c in enumerate(classes):
            means[i] = x[np.where(y == c)].mean(axis=0)
        return means[0], means[1]
    
    def _calc_Cov(self, x, y, mu0, mu1):
        cov0 = np.sum()
    
    def project(self, x_data):
        return np.dot(x_data, self.weights)