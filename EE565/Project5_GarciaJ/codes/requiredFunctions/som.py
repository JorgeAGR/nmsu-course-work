#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:49:42 2019

@author: jorgeagr
"""

import numpy as np


def closest_node(data, t, som, m_rows, m_cols):
  # (row,col) of map node closest to data[t]
  result = (0,0)
  small_dist = 1.0e20
  for i in range(m_rows):
    for j in range(m_cols):
      ed = euc_dist(som[i][j], data[t])
      if ed < small_dist:
        small_dist = ed
        result = (i, j)
  return result

def euc_dist(v1, v2):
  return np.linalg.norm(v1 - v2) 

def manhattan_dist(r1, c1, r2, c2):
  return np.abs(r1-r2) + np.abs(c1-c2)
'''
def most_common(lst, n):
  # lst is a list of values 0 . . n
  if len(lst) == 0: return -1
  counts = np.zeros(shape=n, dtype=np.int)
  for i in range(len(lst)):
    counts[lst[i]] += 1
  return np.argmax(counts)
'''
def SelfOrganizingMap(data, nrows, ncols, maxsteps=5000, learning_rate=0.5, sigma=5):
    '''
    data
    '''
    dim = data.shape[1]
    som = np.random.random_sample(size=(nrows,ncols,dim))
    for s in range(maxsteps):
        pct_left = 1.0 - ((s * 1.0) / maxsteps)
        curr_range = (int)(pct_left * (nrows + ncols))
        curr_rate = pct_left * learning_rate
    
        t = np.random.randint(len(data))
        (bmu_row, bmu_col) = closest_node(data, t, som, nrows, ncols)
        for i in range(nrows):
            for j in range(ncols):
                #h = np.exp(euc_dist(np.asarray([bmu_row, bmu_col]), np.asarray([i,j]))**2 / (2*sigma**2))
                #som[i][j] = som[i][j] + curr_rate * (data[t] - som[i][j])
                if manhattan_dist(bmu_row, bmu_col, i, j) < curr_range:
                    som[i][j] = som[i][j] + curr_rate * (data[t] - som[i][j])
    
    u_matrix = np.zeros(shape=(nrows,ncols), dtype=np.float64)
    for i in range(nrows):
        for j in range(ncols):
            v = som[i][j]  # a vector 
            sum_dists = 0.0; ct = 0
            if i-1 >= 0:    # above
                sum_dists += euc_dist(v, som[i-1][j]); ct += 1
            if i+1 <= nrows-1:   # below
                sum_dists += euc_dist(v, som[i+1][j]); ct += 1
            if j-1 >= 0:   # left
                sum_dists += euc_dist(v, som[i][j-1]); ct += 1
            if j+1 <= ncols-1:   # right
                sum_dists += euc_dist(v, som[i][j+1]); ct += 1
            u_matrix[i][j] = sum_dists / ct
    
    return som, u_matrix