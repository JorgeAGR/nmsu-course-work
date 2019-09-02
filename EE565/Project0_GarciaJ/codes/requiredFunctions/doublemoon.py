#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:27:42 2019

@author: jorgeagr
"""
import numpy as np
from scipy.stats import uniform

def doublemoon(N, w, r, d):
    '''
    N: Number of samples
    w: Moon width
    r: Moon radius
    d: Lower moon distance from x-axis
    '''
    data = np.ones((N,3))
    
    a = r + w/2
    b = r - w/2
    magnitudes = a + (b - a) * uniform.rvs(size=N)
    pos_angles = np.pi * uniform.rvs(size=N//2)
    neg_angles = -np.pi * uniform.rvs(size=N//2)
    data[:N//2,0] = magnitudes[:N//2] * np.cos(pos_angles)
    data[:N//2,1] = magnitudes[:N//2] * np.sin(pos_angles)
    data[N//2:,0] = magnitudes[N//2:] * np.cos(neg_angles) + r
    data[N//2:,1] = magnitudes[N//2:] * np.sin(neg_angles) - d
    data[N//2:,2] = -data[N//2:,2]
    
    return data