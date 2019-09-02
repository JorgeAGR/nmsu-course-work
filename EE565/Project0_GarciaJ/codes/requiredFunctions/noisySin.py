#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 20:06:19 2019

@author: jorgeagr
"""
import numpy as np

def noisySin(N, var):
    data = np.ones((N, 2))
    x = np.sort(np.random.rand(N))
    noise = np.random.normal(scale=np.sqrt(var), size=N)
    t = np.sin(2*np.pi*x) + noise
    data[:,0] = x
    data[:,1] = t
    return data