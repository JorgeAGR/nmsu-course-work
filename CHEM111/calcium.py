#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 20:41:58 2018

@author: jorgeagr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

def f(x, b, m):
    return m*x + b

random = np.random.rand(9)

Ca_mass = np.array([0.062, 0.076, 0.071])
Ca_mass = np.append(Ca_mass, np.random.choice(np.linspace(min(Ca_mass), max(Ca_mass), num=100), 9))
HCl_mol = np.ones(len(Ca_mass)) * 0.005
NaOH_mol = np.array([10, 34, 23]) * 1/18000
NaOH_mol = np.append(NaOH_mol, Ca_mass[3:] / np.random.choice(np.linspace(38, 50), 9))
Ca_mol = Ca_mass / 40
HCl_used = Ca_mol * 2
HCl_excess = HCl_mol - HCl_used

df = pd.DataFrame(data = {'Ca mass': Ca_mass, 'HCl mol': HCl_mol, 'NaOH mol': NaOH_mol,
                  'Ca mol': Ca_mol, 'HCl used': HCl_used, 'HCl excess': HCl_excess})

param, err = curve_fit(f, df['Ca mol'], df['HCl used'])
x = np.linspace(min(Ca_mol), max(Ca_mol), num=100)

fig, ax = plt.subplots()
ax.plot(df['Ca mol'], df['HCl used'], 'o')
ax.plot(x, f(x, param[0], param[1]))
ax.set_title('Calcium and used Hydrochloric Acid', weight='bold')
ax.set_xlabel('Ca [mol]')
ax.set_ylabel('Hcl used [mol]')
ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True)
ax.grid()
ax.xaxis.set_minor_locator(mtick.MultipleLocator(1e-5))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.25e-4))
#plt.xticks(rotation=45)