#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 06:25:42 2020

@author: jorgeagr
"""

import numpy as np

G = np.array([[1, 1, 0],
              [0, 0, 1],
              [0, 0, 1]])
    
d = np.array([[1], [2], [1]])

C = np.eye(3) * np.array([1, 2, 3])

Gnew = G @ np.linalg.inv(C)
Gnewg = np.linalg.pinv(Gnew)

Cm = Gnewg @ d

m = np.linalg.inv(C) @ Cm

print('Minimized weighted length:')
print('Cm = {}'.format(Cm))
print('m = {}'.format(m))