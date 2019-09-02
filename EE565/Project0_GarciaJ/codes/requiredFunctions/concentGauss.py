#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 17:27:15 2019

@author: jorgeagr
"""
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform

def concentGauss(N, r, var_in, var_out):
    data = np.ones((N, 3))
    
    # Inner circular Gaussian points
    data[:N//2,:2] = norm.rvs(size=(N//2, 2)) * np.sqrt(var_in)
    
    outter_mag = norm.rvs(size=N//2) * np.sqrt(var_out) + r
    angles = uniform.rvs(size=N//2) * 2 * np.pi
    #Outter Gaussian annulus points
    data[N//2:,0] = outter_mag * np.cos(angles)
    data[N//2:,1] = outter_mag * np.sin(angles)
    data[N//2:,2] = -data[N//2:,2]
    
    return data