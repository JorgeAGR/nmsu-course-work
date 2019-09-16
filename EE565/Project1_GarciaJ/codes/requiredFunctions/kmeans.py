#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:24:47 2019

@author: jorgeagr
"""

import numpy as np

class KMeans(object):
    
    def fit_batch(self, X_data, k_groups, centroids=None, converge=1e-5, max_iter=200, seed=None):
        '''
        Uses batch training to fit to the entire data set.
        X_data: input data
        k_groups: Number of groups to find
        centroids: Initial centroids chosen, if given. Else, randomly selects points from dataset
        converge: If average change in distance of centroids is equal to this, then exit the routine
        max_iter: Max iterations to run the fit for
        seed: Seed for the RNG when initializing centroids
        '''
        old_centroids = np.zeros((k_groups, X_data.shape[1]))
        if centroids is None:
            np.random.seed(seed)
            centroids = self._shuffle_Data(X_data)[:k_groups]
        #avg_change = np.abs((centroids**2).sum() - (old_centroids**2).sum()).mean()
        avg_change = np.sqrt(((centroids - old_centroids)**2).sum()).mean()
        distancesq = np.zeros((X_data.shape[0], k_groups))
        
        for iters in np.arange(1, max_iter+1, 1):
            for k in range(k_groups):
                mu = np.ones((1, X_data.shape[1])) * centroids[k]
                old_centroids[k] = mu
                distancesq[:, k] = ((X_data - mu)**2).sum(axis=1)
            r_index = self._find_min(distancesq)
            centroids = (np.dot(X_data.T, r_index) / r_index.sum(axis=0)).T
            if np.isnan(centroids).any():
                nan_rows = np.unique(np.argwhere(np.isnan(centroids))[:,0])
                centroids[nan_rows, :] = X_data[np.random.randint(0, X_data.shape[0], len(nan_rows))]
            #avg_change = np.abs((centroids**2).sum() - (old_centroids**2).sum()).mean()
            avg_change = np.sqrt(((centroids - old_centroids)**2).sum()).mean()
            if avg_change <= converge:
                break
        self.centroids = np.sort(centroids, axis=0)
        
        return self.centroids, np.argmax(r_index, axis=1), iters
    
    def fit_online(self, X_data, k_groups, centroids=None, learn_rate=1e-3, converge=1e-5, max_epochs=10, seed=None):
        '''
        Not really stable? Seems to give wild predictions compared to scikit
        Uses online training to fit point by point.
        X_data: input data
        k_groups: Number of groups to find
        centroids: Initial centroids chosen, if given. Else, randomly selects points from dataset
        learn_rate: Coefficient by which centroids are updated
        converge: If average change in distance of centroids is equal to this, then exit the routine
        max_epochs: Max epochs to run the fit for
        seed: Seed for the RNG when initializing centroids
        '''
        old_centroids = np.zeros((k_groups, X_data.shape[1]))
        if centroids is None:
            np.random.seed(seed)
            centroids = self._shuffle_Data(X_data)[:k_groups]
        #avg_change = np.abs((centroids**2).sum() - (old_centroids**2).sum()).mean()
        avg_change = np.sqrt(((centroids - old_centroids)**2).sum()).mean()
        
        for epoch in np.arange(1, max_epochs+1, 1):
            old_centroids[:,:] = centroids
            X_data = self._shuffle_Data(X_data)
            for n in range(X_data.shape[0]):
                point = np.ones((k_groups, X_data.shape[1])) * X_data[n]
                distancesq = ((point - centroids)**2).sum(axis=1).reshape(1, point.shape[0])
                cluster_ind = np.argmax(self._find_min(distancesq))
                centroids[cluster_ind] = centroids[cluster_ind] + learn_rate * (point - centroids[cluster_ind])
            #avg_change = np.abs((centroids**2).sum() - (old_centroids**2).sum()).mean()
            avg_change = np.sqrt(((centroids - old_centroids)**2).sum()).mean()
            if avg_change <= converge:
                break
        self.centroids = np.sort(centroids, axis=0)
        
        return self.centroids, epoch
    
    def predict_cluster(self, X_data, centroids=None):
        '''
        Returns the predicted group the data belongs to, given found centroids
        X_data: input data
        centroids: Centroids of the groups. Typically, the centroids fitted to
        '''
        if centroids is None:
            centroids = self.centroids
        distancesq = np.zeros((X_data.shape[0], centroids.shape[0]))
        for k in range(centroids.shape[0]):
            distancesq[:, k] = ((X_data - centroids)**2).sum(axis=1)
        r_index = self._find_min(distancesq)
        
        return np.argmax(r_index, axis=1)
    
    def eval_cost(self, X_data, centroids=None):
        '''
        Evaluates the cost function given a set of data and found centroids.
        X_data: input data
        centroids: Centroids of the groups. Typically, the centroids fitted to
        '''
        if centroids is None:
            centroids = self.centroids
        k_groups = centroids.shape[0]
        distancesq = np.zeros((X_data.shape[0], k_groups))
        for k in range(k_groups):
            mu = np.ones((1, X_data.shape[1])) * centroids[k]
            distancesq[:, k] = ((X_data - mu)**2).sum(axis=1)
        r_index = self._find_min(distancesq)
        
        return (distancesq * r_index).sum()
    
    def _find_min(self, distancesq):
        '''
        Private function. Finds the centroid with minimum distance squared to a set of data
        '''
        min_ind = distancesq.argmin(axis=1)
        r_index = distancesq * 0
        r_index[np.arange(0, r_index.shape[0], 1), min_ind] = 1
        
        return r_index
    
    def _shuffle_Data(self, X_data):
        '''
        Private function. Randomly shuffles the given data
        '''
        data_ind = np.arange(0, X_data.shape[0], 1)
        np.random.shuffle(data_ind)
        return X_data[data_ind]