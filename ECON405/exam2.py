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
# a
model_1 = regression_model()
model_1.load_csv('wage_ed.csv')
model_1.set_vars('WAGE', 'YRS')
model_1.fit_model()
b0_1 = model_1.params[0]
b1_1 = model_1.params[1]
func1 = lambda x: b0_1 + b1_1 * x

model_1.data.loc[:, 'YRS^2'] = model_1.data.loc[:, 'YRS']**2
model_1.set_vars('WAGE', 'YRS^2')
model_1.fit_model()
b0_2 = model_1.params[0]
b1_2 = model_1.params[1]
func2 = lambda x: b0_2 + b1_2 * x**2

model_1.data.loc[:, 'lnWAGE'] = np.log(model_1.data.loc[:, 'WAGE'])
model_1.data.loc[:, 'lnYRS'] = np.log(model_1.data.loc[:, 'YRS'])
model_1.set_vars('lnWAGE', 'lnYRS')
model_1.fit_model()
b0_3 = model_1.params[0]
b1_3 = model_1.params[1]
func3 = lambda x: np.exp(b0_3) * x ** b1_3
'''
x_grid = np.linspace(min(model_1.data['YRS']), max(model_1.data['YRS']))
fig, ax = plt.subplots()
ax.scatter(model_1.data['YRS'], model_1.data['WAGE'], marker='.', color='black')
ax.plot(x_grid, func1(x_grid), '--', color='gray')
ax.plot(x_grid, func2(x_grid), color='red')
ax.plot(x_grid, func3(x_grid), '--', color='green')
'''

# == Part 3 == #
model_3 = regression_model()
model_3.load_csv('gdp.csv')
model_3.set_vars('RINVEST', 'RAAA')
model_3.fit_model()
model_3.set_vars('RINVEST', 'RGDP')
model_3.fit_model()
model_3.data.loc[:, 'STD_RINVEST'] = (model_3.data.loc[:, 'RINVEST'] - model_3.data['RINVEST'].mean()) / model_3.data['RINVEST'].std()
model_3.data.loc[:, 'STD_RAAA'] = (model_3.data.loc[:, 'RAAA'] - model_3.data['RAAA'].mean()) / model_3.data['RAAA'].std()
model_3.data.loc[:, 'STD_RGDP'] = (model_3.data.loc[:, 'RGDP'] - model_3.data['RGDP'].mean()) / model_3.data['RGDP'].std()
model_3.set_vars('STD_RINVEST', 'STD_RAAA')
model_3.fit_model()
model_3.set_vars('STD_RINVEST', 'STD_RGDP')
model_3.fit_model()
model_3.set_vars('RGDP', 'RCONS', 'RINVEST', 'RGOV', 'RNEX', 'SDISCREP')
model_3.fit_model()


# == Part 4 == #
model_4 = regression_model()
model_4.load_csv('demand_data.csv')
model_4.set_vars('Chicken', 'RPC', 'RPB', 'RDPI')
model_4.fit_model()
print(model_4.params[1] * model_4.data['Chicken'].mean() / model_4.data['RPC'].mean())
model_4.data.loc[:, 'STD_Chicken'] = (model_4.data.loc[:, 'Chicken'] - model_4.data.loc[:, 'Chicken'].mean()) / model_4.data.loc[:, 'Chicken'].std()
model_4.data.loc[:, 'STD_RPC'] = (model_4.data.loc[:, 'RPC'] - model_4.data.loc[:, 'RPC'].mean()) / model_4.data.loc[:, 'RPC'].std()
model_4.data.loc[:, 'STD_RPB'] = (model_4.data.loc[:, 'RPB'] - model_4.data.loc[:, 'RPB'].mean()) / model_4.data.loc[:, 'RPB'].std()
model_4.data.loc[:, 'STD_RDPI'] = (model_4.data.loc[:, 'RDPI'] - model_4.data.loc[:, 'RDPI'].mean()) / model_4.data.loc[:, 'RDPI'].std()
model_4.set_vars('STD_Chicken', 'STD_RPC', 'STD_RPB', 'STD_RDPI')
model_4.fit_model()
print(model_4.params[1] * model_4.data['STD_Chicken'].mean() / model_4.data['STD_RPC'].mean())


# == Part 5 == #
model_5 = regression_model()
model_5.load_csv('teacher_salary.csv')
model_5.set_vars('SALARY', 'SPEND')
model_5.fit_model()
model_5.set_vars('SALARY', 'SPEND', 'D1', 'D2')
model_5.fit_model()
model_5.data.loc[:, 'SPEND*D1'] = model_5.data.loc[:, 'SPEND'] * model_5.data.loc[:, 'D1']
model_5.data.loc[:, 'SPEND*D2'] = model_5.data.loc[:, 'SPEND'] * model_5.data.loc[:, 'D2']
model_5.set_vars('SALARY', 'SPEND', 'SPEND*D1', 'SPEND*D2')
model_5.fit_model()