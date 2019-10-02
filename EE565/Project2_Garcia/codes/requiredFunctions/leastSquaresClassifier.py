# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:48:12 2019

@author: jorge
"""

import numpy as np
from oneHotEncode import oneHotEncode

class LeastSquares_Classifier(object):
    
    def __init__(self, train_x, train_y):
        train_y = oneHotEncode(train_y)
        train_x = np.hstack((np.ones((len(train_x),1)), train_x))
        
        self.fit(train_x, train_y)
        
        return
    
    def fit(self, train_x, train_y):
        self.weights = np.dot(np.linalg.pinv(train_x), train_y)
        return self.weights
    
    def predict(self, x_data):
        x_data = np.hstack((np.ones((len(x_data), 1)), x_data))
        return np.dot(x_data, self.weights)