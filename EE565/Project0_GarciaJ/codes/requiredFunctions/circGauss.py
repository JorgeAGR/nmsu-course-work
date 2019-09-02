#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:33:18 2019

@author: jorgeagr
"""

import numpy as np
from scipy.stats import norm

def circGauss(N, var, *mus):
    '''
    N: Number of points to be sampled
    var: Variance of all distribution
    mus: Means of the various dimensions (also tells dimensionality)
    '''
    data = norm.rvs(size=(N, len(mus))) * np.sqrt(var) + mus
    
    return data