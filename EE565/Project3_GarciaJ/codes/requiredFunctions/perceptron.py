#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:07:20 2019

@author: jorgeagr
"""

import numpy as np

def perceptron(x, w):
    '''
    x: Data matrix in column form. Rows are attributes, columns are instances.
    w: Column vector, each row is weight of attribute.
    '''
    y = np.dot(w.T, x)
    
    y[y > 0] = 1
    y[y < 0] = -1
    
    return y.flatten()