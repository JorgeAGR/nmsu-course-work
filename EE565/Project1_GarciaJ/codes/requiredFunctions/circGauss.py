#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:33:18 2019

@author: jorgeagr
"""

import numpy as np

def circGauss(N, var, *mus, seed=None):
    '''
    N: Number of points to be sampled
    var: Variance of all distribution
    mus: Means of the various dimensions (also tells dimensionality)
    seed: RNG seed
    '''
    np.random.seed(seed)
    data = np.random.normal(loc=mus, scale=np.sqrt(var), size=(N, len(mus)))
    return data