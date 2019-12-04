#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:35:49 2019

@author: jorgeagr
"""

import numpy as np

def histogram(data, n_bins):
    xmin = data.min()
    xmax = data.max()
    b = np.linspace(xmin, xmax, n_bins)
    h = np.zeros(n_bins)
    for m in range(len(data)):
        e = np.abs(b - data[m])
        i = e.argmin()
        h[i] += 1
    return h, b