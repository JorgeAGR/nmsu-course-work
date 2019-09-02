#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:55:56 2019

@author: jorgeagr
"""
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform

def gaussX(N, var):
    data = np.ones((N, 3))
    data[:,:2] = norm.rvs(size=(N, 2)) * np.sqrt(var)
    data[:,2] = -data[:,0] * data[:,1] / np.abs(data[:,0] * data[:,1])
    
    return data