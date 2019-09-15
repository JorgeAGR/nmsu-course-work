#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 17:27:15 2019

@author: jorgeagr
"""
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform

def concentGauss(N, r, var_in, var_out, seed=None):
    '''
    N: Number of samples
    r: Gaussian annulus mean radius
    var_in: Inner gaussian variance
    var_out: Outer gaussian variance
    seed: RNG seed
    '''
    np.random.seed(seed)
    
    data = np.ones((N, 3))

    # Inner circular Gaussian points
    data[:N//2,:2] = np.random.normal(scale=np.sqrt(var_in), size=(N//2, 2))
    
    #Outter Gaussian annulus points
    outter_mag = np.random.normal(loc=r, scale=np.sqrt(var_out), size=(N//2))
    angles = np.random.uniform(high=2*np.pi, size=N//2)
    data[N//2:,0] = outter_mag * np.cos(angles)
    data[N//2:,1] = outter_mag * np.sin(angles)
    data[N//2:,2] = -data[N//2:,2]
    
    return data