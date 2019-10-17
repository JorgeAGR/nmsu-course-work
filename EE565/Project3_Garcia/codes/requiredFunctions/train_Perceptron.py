#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:40:00 2019

@author: jorgeagr
"""

import numpy as np
from perceptron import perceptron


class TrainPerceptron():
    
    def __init__(self):
        self.error_log = []
        return

    def fit_online(self, x_data, y_data, learning_rate=1e-3, threshold=1e-5, max_epochs=100, seed=None):
        '''
        x_data: Data matrix in row form. Rows are instances, cols are attributes.
        y_data: Array of labels
        '''
        np.random.seed()
        self.weights = np.zeros((x_data.shape[1], 1))
        x = x_data.T
        
        data_ind = np.arange(0, x_data.shape[1])
        for epoch in range(max_epochs):
            w_old = self.weights
            np.random.shuffle(data_ind)
            x_data = x_data[data_ind]
            y_data = y_data[data_ind]
            for n in range(x_data.shape[1]):
                y_pred = perceptron(x[:,n], self.weights)
                e = y_data[n] - y_pred
                self.weights = self.weights + learning_rate * e * x[:,n]
            if np.mean(np.abs(self.weights - w_old)) < threshold:
                break
    
    def score(self, x_data):
        y_pred = perceptron(x_data, self.weights)
        return