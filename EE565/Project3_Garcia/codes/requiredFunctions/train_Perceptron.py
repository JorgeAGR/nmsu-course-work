#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:40:00 2019

@author: jorgeagr
"""

import numpy as np
from requiredFunctions.perceptron import perceptron

class PerceptronClassifier(object):
    
    def __init__(self):
        self.accuracy_log = []
        self.cost_log = []
        return

    def fit_Online(self, x_data, y_data, w0=None, learning_rate=1e-3, threshold=1e-5, max_epochs=100, seed=None):
        '''
        x_data: Data matrix in row form. Rows are instances, cols are attributes.
        y_data: Array of labels
        '''
        np.random.seed(seed)
        weights_shape = (x_data.shape[1] + 1, 1)
        if w0 is None:
            self.weights = np.zeros(weights_shape)
        else:
            self.weights = w0.reshape(weights_shape)
        data_ind = np.arange(0, x_data.shape[0])
        x_data = np.hstack([np.ones((x_data.shape[0], 1)), x_data])
        x_data = x_data.T
        for epoch in range(max_epochs):
            w_old = self.weights
            np.random.shuffle(data_ind)
            x_data = x_data[:,data_ind]
            y_data = y_data[data_ind]
            for n in range(x_data.shape[1]):
                y_pred = perceptron(x_data[:,n], self.weights)
                e = y_data[n] - y_pred
                self.weights = self.weights + (learning_rate * e * x_data[:,n]).reshape(weights_shape)
                self.accuracy_log.append(self._fit_Score(x_data, y_data))
            if np.mean(np.abs(self.weights - w_old)) < threshold:
                break
        self.accuracy_log = np.asarray(self.accuracy_log)
        return self.weights
    
    def fit_Batch(self, x_data, y_data, learning_rate=1e-3, threshold=1e-5, max_epochs=100, seed=None):
        np.random.seed(seed)
        weights_shape = (x_data.shape[1] + 1, 1)
        self.weights = np.random.rand(x_data.shape[1] + 1,1) - 0.5
        x_data = np.hstack([np.ones((x_data.shape[0], 1)), x_data])
        x_data = x_data.T
        for epoch in range(max_epochs):
            w_old = self.weights
            y_pred = perceptron(x_data, self.weights)
            wrong = y_pred != y_data
            #self.weights = self.weights + (learning_rate * x_data.T[wrong].T.sum(axis=1)).reshape(weights_shape)
            self.weights = self.weights + (learning_rate * (x_data[:,wrong] * y_data[wrong]).sum(axis=1)).reshape(weights_shape)
            self.accuracy_log.append(self._fit_Score(x_data, y_data))
            self.cost_log.append(self.cost_Batch(x_data, y_data))
            if np.mean(np.abs(self.weights - w_old)) < threshold:
                    break
        self.accuracy_log = np.asarray(self.accuracy_log)
        self.cost_log = np.asarray(self.cost_log)
        return self.weights
    
    def predict(self, x_data):
        x_data = np.hstack([np.ones((x_data.shape[0], 1)), x_data])
        x_data = x_data.T
        return perceptron(x_data, self.weights)
    
    def _fit_Score(self, x_data, y_data):
        y_pred = perceptron(x_data, self.weights)
        diff = y_data - y_pred
        diff[diff!=0] = 1
        error = diff.sum() / len(y_data)
        return 1 - error
    
    def cost_Batch(self, x_data, y_data):
        y_pred = perceptron(x_data, self.weights)
        wrong = y_pred != y_data
        return (np.dot(self.weights.T, x_data.T[wrong].T)**2).sum()