#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:35:13 2019

@author: jorgeagr
"""

import numpy as np

def euclidian(x_arr):
    return np.sqrt((x_arr**2).sum(axis=1))

