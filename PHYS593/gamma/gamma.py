#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:17:26 2018

@author: jorgeagr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:17:32 2018

@author: jorgeagr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.stats import chi2
import matplotlib.ticker as mtick

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)


def latex_table(angles, counts, sample_time, right_singles, left_singles, filename):
    
    df = pd.DataFrame(data={'Angle [deg]': angles, 'Coincidence Counts [counts/Sample Time]': counts, 
          'Sample Time [s]': sample_time, 'Right Detector Singles [counts/s]': right_singles,
          'Left Detector Singles [counts/s]': left_singles})
    
    with open(filename, 'w+') as file:
        print(df.to_latex(), file=file)

def analyze(func, angles, counts, right_singles, left_singles, sample_time, x, fit='regular'):
    # x is just grid to evaluate fitted function
    fracc_counts = (np.sqrt(counts)/sample_time) / (counts/sample_time)
    fracc_right_singles = np.sqrt(right_singles)/right_singles
    fracc_left_singles = np.sqrt(left_singles)/left_singles
    
    norm_counts = counts / sample_time /(right_singles * left_singles)
    norm_counts_err = norm_counts * np.sqrt(fracc_counts**2 + fracc_right_singles**2 + fracc_left_singles**2)
    
    if fit == 'regular':
        param, error = curve_fit(func, angles, norm_counts, sigma=norm_counts_err)
    else:
        param, error = curve_fit(func, angles, norm_counts, sigma=norm_counts_err,
                                 p0=[max(norm_counts), np.mean(norm_counts), np.std(norm_counts)])
    print(param)
    
    if fit == 'regular':
        chisq = np.sum(( (func(angles, param[0]) - norm_counts) / norm_counts_err ) ** 2)
    else:
        chisq = np.sum(( (func(angles, param[0], param[1], param[2]) - norm_counts) / norm_counts_err ) ** 2)
    
    rechisq = chisq / (len(counts) - len(param))
    
    return {'param': param, 'error': error, 'chisq': chisq, 'rechisq': rechisq,
            'norm counts': norm_counts, 'norm counts err': norm_counts_err}


def cobalt_60():
    def f(theta, amp):
        theta = np.deg2rad(theta)
        return amp * (1 + (1/8) * (np.cos(theta)) ** 2 + (1/24) * (np.cos(theta)) ** 4)
    
    sample_time = 600 # seconds
    angles = np.array([180, 170, 160, 150, 140, 130, 120, 110, 100, 90]) # degrees
    counts = np.array([654, 744, 788, 737, 705, 695, 754, 716, 720, 719]) # counts per 10 min
    right_singles = np.array([915, 915, 887, 965, 912, 943, 918, 931, 958, 909]) # Hz
    left_singles = np.array([877, 916, 836, 908, 806, 910, 924, 984, 826, 1061])
    resolving_time = 55e-9 # s
    resolving_time_err = 0.3e-9 # s
    
    x = np.arange(90, 190, 1)
    
    #print(analyze(f, angles, counts, right_singles, left_singles, sample_time, x))
    
    fracc_counts = (np.sqrt(counts)/sample_time) / (counts/sample_time)
    fracc_right_singles = np.sqrt(right_singles)/right_singles
    fracc_left_singles = np.sqrt(left_singles)/left_singles
    fracc_resolving_time = resolving_time_err / resolving_time
    
    random_coincidences = left_singles * right_singles * resolving_time
    random_coincidences_err = random_coincidences * np.sqrt( fracc_left_singles**2 + fracc_right_singles**2 + fracc_resolving_time**2 )
    
    norm_counts = counts / sample_time /(right_singles * left_singles)
    norm_counts_err = norm_counts * np.sqrt(fracc_counts**2 + fracc_right_singles**2 + fracc_left_singles**2)
    
    
    norm_counts_err = norm_counts_err / min(norm_counts)
    norm_counts = norm_counts / min(norm_counts)
    
    param, error = curve_fit(f, angles, norm_counts, sigma=norm_counts_err)
    error = np.sqrt(np.diag(error))
    
    print('Coefficients:', param)
    print('Errors:', error)
    
    dof = len(counts) - len(param)
    chisq = np.sum(( (f(angles, *param) - norm_counts) / norm_counts_err ) ** 2)
    rechisq = chisq / dof
    pval = chi2.sf(chisq, dof) * 100
    
    print('ChiSq:', chisq)
    print('dof:', dof)
    print('ChiSq per dof:', rechisq)
    print('p-value:', pval, '%')
    
    latex_table(angles, counts, sample_time, right_singles, left_singles, 'co60_table.txt')
    
    figrand, axrand = plt.subplots()
    axrand.errorbar(angles, random_coincidences, yerr = random_coincidences_err,
                    fmt='.', color='black', label='Random Coincidences')
    axrand.set_xlabel('Angle [deg]')
    axrand.set_ylabel('Counts [1/s]')
    axrand.xaxis.set_major_locator(mtick.MultipleLocator(10))
    axrand.xaxis.set_minor_locator(mtick.MultipleLocator(1))
    axrand.yaxis.set_minor_locator(mtick.MultipleLocator(0.0005))
    axrand.yaxis.set_major_locator(mtick.MultipleLocator(0.0025))
    axrand.set_ylim(0.0380,0.0560)
    axrand.set_xlim(87, 183)
    axrand.grid()
    axrand.legend(loc='upper left')
    plt.tight_layout()
    
    x = np.linspace(90, 180, num=100)
    fig, ax = plt.subplots()
    ax.errorbar(angles, norm_counts, yerr=norm_counts_err, fmt='.', 
                color='black', label='Data')
    #ax.plot(angles, f(angles, *param))
    ax.plot(x, f(x, *param), color='gray', label='Fit')
    ax.set_xlabel(r'Angle [deg]')
    ax.set_ylabel('Normalized Counts [1/s]')
    ax.xaxis.set_major_locator(mtick.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(2))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.02))
    ax.set_ylim(0.9,)
    ax.set_xlim(86, 184)
    ax.grid()
    ax.legend()
    plt.tight_layout()

def sodium_22(): 
    
    def f(x, A, mu, std, c0):
        return A * np.exp(-(x - mu)**2 / (2 * std ** 2)) + c0 #+ c1 * x
    
    #def f(x, A, mu, std, B, k):
    #    return A * np.exp(-(x - mu)**2 / (2 * std ** 2)) + B * np.exp(-k * x)
    
    sample_time =   np.array([600, 600, 600, 600, 600, 600, 600, 600,  600,   60,   60,  60,   60,   60,   60,   60,  60,  60,  60]) # seconds
    angles =        np.array([ 90, 100, 110, 120, 130, 140, 150, 160,  170,  180,  190, 175,  185,  177,  183,  184, 200, 210, 220,]) # degrees
    counts =        np.array([839, 701, 684, 639, 649, 719, 687, 721, 3367, 1583, 1206, 957, 1843, 1280, 1901, 1891, 159,  65,  84,]) # counts per 10 min
    right_singles = np.array([513, 503, 564, 544, 530, 512, 514, 491,  535,  563,  551, 538,  522,  597,  547,  558, 535, 550, 643,]) # Hz
    left_singles =  np.array([236, 248, 265, 299, 278, 300, 284, 293,  301,  315,  286, 299,  305,  308,  316,  301, 301, 299, 288,])
    
    sort_ind = np.argsort(angles)
    start = 6
    
    angles = angles[sort_ind][start:]
    sample_time = sample_time[sort_ind][start:]
    counts = counts[sort_ind][start:]
    right_singles = right_singles[sort_ind][start:]
    left_singles = left_singles[sort_ind][start:]
    resolving_time = 55e-9 # s
    resolving_time_err = 3e-9 # s
    #y = f(x, 5) + np.random.random((len(x))) * 10
    
    #fit = analyze(f, angles, counts, right_singles, left_singles, sample_time, x, fit='normal')
    
    fracc_counts = (np.sqrt(counts)/sample_time) / (counts/sample_time)
    fracc_right_singles = np.sqrt(right_singles) / right_singles
    fracc_left_singles = np.sqrt(left_singles) / left_singles
    fracc_resolving_time = resolving_time_err / resolving_time
    
    random_coincidences = left_singles * right_singles * resolving_time
    random_coincidences_err = random_coincidences * np.sqrt( fracc_left_singles**2 + fracc_right_singles**2 + fracc_resolving_time**2 )
    
    norm_counts = counts / sample_time /(right_singles * left_singles)
    norm_counts_err = norm_counts * np.sqrt(fracc_counts**2 + fracc_right_singles**2 + fracc_left_singles**2)
    
    norm_counts_err = norm_counts_err / max(norm_counts)
    norm_counts = norm_counts / max(norm_counts)
    
    mean = np.sum(angles * norm_counts) / np.sum(norm_counts)
    sqmean = np.sum(angles ** 2 * norm_counts) / np.sum(norm_counts)
    std = np.sqrt(sqmean - mean**2)
    
    slope_guess = (norm_counts[-1] - norm_counts[0]) / (angles[-1] - angles[0])
    decay_guess = 0.0001
    
    param, error = curve_fit(f, angles, norm_counts, sigma=norm_counts_err,
                             p0=[max(norm_counts), mean, std, 
                                 norm_counts[0]])
    error = np.sqrt(np.diag(error))
    
    print('Coefficients:', param)
    print('Errors:', error)
    
    dof = len(counts) - len(param)
    chisq = np.sum(( (f(angles, *param) - norm_counts) / norm_counts_err ) ** 2)
    rechisq = chisq / dof
    pval = chi2.sf(chisq, dof) * 100
    
    print('ChiSq:', chisq)
    print('dof:', dof)
    print('ChiSq per dof:', rechisq)
    print('p-value:', pval, '%')
    
    latex_table(angles, counts, sample_time, right_singles, left_singles, 'na22_table.txt')
    
    figrand, axrand = plt.subplots()
    axrand.errorbar(angles, random_coincidences, yerr = random_coincidences_err,
                    fmt='.', color='black', label='Random Coincidences')
    axrand.set_xlabel('Angle [deg]')
    axrand.set_ylabel('Counts [1/s]')
    axrand.xaxis.set_major_locator(mtick.MultipleLocator(10))
    axrand.xaxis.set_minor_locator(mtick.MultipleLocator(1))
    axrand.yaxis.set_minor_locator(mtick.MultipleLocator(0.0001))
    axrand.set_ylim(0.0070,0.0110)
    axrand.set_xlim(147, 223)
    axrand.grid()
    axrand.legend(loc='upper left')
    plt.tight_layout()
    
    x = np.linspace(150, 220, num=100)
    fig, ax = plt.subplots()
    ax.errorbar(angles, norm_counts, yerr=norm_counts_err, fmt='.', 
                color='black', label='Data')
    #ax.plot(angles, f(angles, *param))
    ax.plot(x, f(x, *param), color='gray', label = 'Fit')
    ax.set_xlabel('Angle [deg]')
    ax.set_ylabel('Normalized Counts [1/s]')
    ax.xaxis.set_major_locator(mtick.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.02))
    ax.set_ylim(0,)
    ax.grid()
    ax.legend()
    plt.tight_layout()