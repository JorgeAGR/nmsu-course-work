# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:38:04 2019

@author: jorge
"""
import numpy as np

class FishersDiscriminant(object):
    
    def __init__(self, train_x, train_y):
        
        train_x = train_x.T
        
        mu0, mu1 = self._calc_Means(train_x, train_y)
        cov = self._calc_Cov(train_x, train_y, mu0, mu1)
        self._calc_Weights(mu0, mu1, cov)
        self.discriminant = self.project(train_x.T).mean()# - self.weights[0]
        return
    
    def _calc_Weights(self, mu0, mu1, cov):        
        w = np.dot(np.linalg.inv(cov), (mu1 - mu0))
        w_mag = np.sqrt((w**2).sum())
        self.weights = (w / w_mag)
        return self.weights
    
    def _calc_Means(self, x, y):
        classes = np.unique(y)
        self.means = np.zeros((x.shape[0], len(classes)))
        for i, c in enumerate(classes):
            self.means[:,i] = x[:,y == c].mean(axis=1)
        mu0 = self.means[:,0].reshape(self.means.shape[0],1)
        mu1 = self.means[:,1].reshape(self.means.shape[0],1)
        return mu0, mu1
    
    def _calc_Cov(self, x, y, mu0, mu1):        
        classes = np.unique(y)
        cov0 = np.dot((x[:,y == classes[0]] - mu0), (x[:,y == classes[0]] - mu0).T)
        cov1 = np.dot((x[:,y == classes[1]] - mu1), (x[:,y == classes[1]] - mu1).T)
        return cov0 + cov1
    
    def project(self, x_data):
        return np.dot(x_data, self.weights)
    
    def predict(self, x_data):
        projection = self.project(x_data)
        return (projection >= self.discriminant).astype(np.int).flatten()
    
    def score(self, x_data, y_data):
        pred = self.predict(x_data)
        wrong = np.abs(y_data - pred)
        err = np.zeros_like(wrong)
        err[wrong > 0] = 1
        accuracy = 1 - np.sum(err)/len(y_data)
        return accuracy