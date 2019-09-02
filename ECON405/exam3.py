# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:44:31 2019

@author: jorge
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2
import sklearn.linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
from regressor import regression_model

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

# == Part 1 == #
model_1 = regression_model()
model_1.load_csv('home_sales.csv')
model_1.set_vars('SALES', 't')
model_1.fit_model()
b0_1 = model_1.params[0]
b1_1 = model_1.params[1]
func1 = lambda x: b0_1 + b1_1 * x

model_1.set_vars('SALES', 't', 'D')
model_1.fit_model()
b0_2 = model_1.params[0]
b1_2 = model_1.params[1]
b2_2 = model_1.params[2]
func2_1 = lambda x: b0_2 + b1_2 * x
func2_2 = lambda x: b0_2 + b2_2 + b1_2 * x

x_grid = np.linspace(min(model_1.data['t']), max(model_1.data['t']), num=500)
fig, ax = plt.subplots()
ax.scatter(model_1.data['t'], model_1.data['SALES'], marker='.', color='black')
ax.plot(x_grid, func1(x_grid), '--', color='gray')
ax.plot(x_grid, func2_1(x_grid), '--', color='red')
ax.plot(x_grid, func2_2(x_grid), '--', color='red')
plt.close()


# == Part 1 == #
model_2 = regression_model()
model_2.load_csv('epe_sales_cus.csv')
model_2.set_vars('AHEC', 't', 'D1', 'D2')
model_2.fit_model()
b0_3 = model_2.params[0]
b1_3 = model_2.params[1]
b2_3 = model_2.params[2]
b3_3 = model_2.params[3]
func3_1 = lambda x: b0_3 + b1_3 * x # D1 = D2 = 0
func3_2 = lambda x: b0_3 + b2_3 + b1_3 * x # D1 = 1 - Dec or Jan
func3_3 = lambda x: b0_3 + b3_3 + b1_3 * x # D2 = 1 - June, July or Aug

x_grid = np.linspace(min(model_2.data['t']), max(model_2.data['t']), num=500)
fig, ax = plt.subplots()
ax.scatter(model_2.data['t'], model_2.data['AHEC'], marker='.', color='black')
ax.plot(x_grid, func3_1(x_grid), '--', color='gray')
ax.plot(x_grid, func3_2(x_grid), '--', color='gray')
ax.plot(x_grid, func3_3(x_grid), '--', color='gray')
plt.close()