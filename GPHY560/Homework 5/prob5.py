#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 05:02:37 2020

@author: jorgeagr
"""

import numpy as np

U = np.array([[.5, -.71, -.5],
              [.5, .71, -.5],
              [.71, 0, .71]])
S = np.array([[1.8, 0, 0, 0],
              [0, 1.4, 0, 0],
              [0, 0, .77, 0]])
V = np.array([[.65, .27, .65, .27],
               [-.5, -.5, .5, .5],
               [.27, -.65, .27, -.65],
               [-.5, .5, .5, -.5]]).T

Up = U
Sp = S[:,:-1]
Vp = V[:,:3]

Gg = Vp @ np.linalg.inv(Sp) @ Up.T

d = np.array([[1], [0], [0]])

m = Gg @ d