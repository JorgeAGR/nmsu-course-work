#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:55:56 2019

@author: jorgeagr
"""
import numpy as np

def gaussX(N, var, seed=None):
    '''
    N: Number of samples
    var: Distribution variance
    seed: RNG seed
    '''
    np.random.seed(seed)
    data = np.ones((N, 3))
    data[:,:2] = np.random.normal(scale=np.sqrt(var), size=(N, 2))
    data[:,2] = -data[:,0] * data[:,1] / np.abs(data[:,0] * data[:,1])
    
    return data