# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:17:00 2019

@author: jorge
"""

import numpy as np
from requiredFunctions.train_Perceptron import PerceptronClassifier
from requiredFunctions.doubleMoon import doubleMoon
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

height = 10
width = 10

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.titlesize'] = 'small'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 24

N = 500
r = 1
w = 0.6
d  = [0.5, 0, -0.5]
data = doubleMoon(N, w, r, d[0], seed=0)
x_train, y_train = data[:,:2], data[:,2]

perceptron = PerceptronClassifier()
perceptron.fit_online(x_train, y_train, seed=0)