#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:39:17 2018

@author: jorgeagr
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit
from scipy.stats import chi2

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

'''
All energies are in KeV
'''
# Source: Filedir, Cutoff
sources = {'Na22': ('data/Na-22GamCorr.TKA', 225, (442.169, 904.662)),}
file, cutoff, peaks = sources['Na22']

def plotData(file, peaks, cutoff=0):
    
    def gaussian(x, A, mu, std):
        return A * np.exp(-(x - mu)**2 / (2 * std ** 2))
    
    with open(file) as data:
         lines = data.readlines()
         for i, line in enumerate(lines):
             lines[i] = int(line.rstrip('\n'))    
    lines = np.array(lines)
    channels = np.arange(1, len(lines)+1)
    
    lines = lines[cutoff:]
    channels = channels[cutoff:]
    
    fig, ax = plt.subplots()
    ax.scatter(channels, lines, marker='.', color='black')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Counts [1/s]')
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
    ax.set_xlim(220, 1030)
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(5))
    ax.text(470, 180, '0.511 MeV', weight='semibold')
    ax.text(930, 60, '1.274 MeV', weight='semibold')
    
    for n in peaks:
        ax.axvline(n, color='red', linestyle='--')
    
    mu1 = peaks[0]
    std1 = 59.7951 / 2 / np.sqrt(2 * np.log(2))
    ch1 = np.arange(-3.5*std1, 3.5*std1) + mu1
    ax.plot(ch1, gaussian(ch1, 150.696, mu1, std1), '--', color='gray')
    
    mu2 = peaks[1]
    std2 = 46.9399 / 2 / np.sqrt(2 * np.log(2))
    ch2 = np.arange(-3.5*std2, 3.5*std2) + mu2
    ax.plot(ch2, gaussian(ch2, 36.681, mu2, std2), '--', color='gray')
    
    plt.tight_layout()

file1 = 'data/40 degrees.TKA'
file2 = 'data/60 degrees.TKA'
file3 = 'data/180 degrees.TKA'
def plotData2(file1, file2, cutoff=10):
    counts = []
    channels = []
    for i, file in enumerate((file1, file2)):
        with open(file) as data:
            lines = data.readlines()
            for i, line in enumerate(lines):
                lines[i] = int(line.rstrip('\n'))    
        lines = np.array(lines)
        chs = np.arange(1, len(lines)+1)
        
        lines = lines[cutoff:]
        chs = chs[cutoff:]
        counts.append(lines)
        channels.append(chs)
    
    files = (file1, file2)
    fig, axes = plt.subplots(nrows=2, sharex=True)
    for i, ax in enumerate(axes):
        title = files[i].split('/')[1].split(' ')[0] + ' deg'
        ax.set_title(title)
        ax.scatter(channels[i], counts[i], marker='.', color='black')
        if i != 0:
            ax.set_xlabel('Channel')
        ax.set_ylabel('Counts [1/s]')
        ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 15)
        ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
        ax.yaxis.set_minor_locator(mtick.MultipleLocator(1))
    plt.tight_layout()

def ch2energy():
    
    quad = False
    
    def linear(x, c0, c1):
        return c0 + c1 * x
    
    def quadratic(x, c0, c1, c2):
        return c0 + c1 * x + c2 * x ** 2
    
    #BaChannels = [56.8473, 137.406, 252.786, 310.95]
    #BaChannels_s = [0.143134, 1.03176, 0.664049, 0.344812]
    BaChannels = [59.8303, 130.943, 247.265, 306.127]
    BaChannels_s = [0.229, 2.077, 2.315, 1.276]
    BaEnergies = [79.5, 160, 302, 367.5]
    BaChiSq = 9357.49
    BaDof = 414 - 19
    BaReChiSq = BaChiSq / BaDof
    Bapval = chi2.sf(BaChiSq, BaDof) * 100
        
    
    #CoChannels = [869.389, 924.018]
    #CoChannels_s = [0.70438, 0.717086]
    if quad == True:
        CoChannels = [869.503, 924.004]
        CoChannels_s = [0.700, 0.712]
        CoEnergies = [1170, 1330]
        CoChiSq = 768.155
        CoDof = 776 - 14
        CoReChiSq = CoChiSq / CoDof
        Copval = chi2.sf(CoChiSq, CoDof) * 100
    
    #Cut the above
    
    #NaChannels = [442.169, 904.662] # Old values
    #NaChennls_s = [0.40892, 0.790727] # Old values
    NaChannels = [442.17, 905.149]
    NaChannels_s = [0.426, 0.934]
    NaEnergies = [511, 1274.53]
    if quad == False:
        del NaChannels[1], NaChannels_s[1], NaEnergies[1]
    NaChiSq = 915.5
    NaDof = 795 - 17
    NaReChiSq = NaChiSq / NaDof
    Napval = chi2.sf(NaChiSq, NaDof) * 100
    #Cut the 2nd energy
    
    #CsChannels = [572.067]
    #CsChannels_s = [0.221484]
    CsChannels = [572.417]
    CsChannels_s = [0.229]
    CsEnergies = [661]
    CsChiSq = 357.104
    CsDof = 330 - 5
    CsReChiSq = CsChiSq / CsDof
    Cspval = chi2.sf(CsChiSq, CsDof) * 100
    
    if quad == True:
        channels = np.hstack([BaChannels, NaChannels, CsChannels, CoChannels,])
        channels_s = np.hstack([BaChannels_s, NaChannels_s, CsChannels_s, CoChannels_s,])
        energies = np.hstack([BaEnergies, NaEnergies, CsEnergies, CoEnergies,])
    else:
        channels = np.hstack([BaChannels, NaChannels, CsChannels,])
        channels_s = np.hstack([BaChannels_s, NaChannels_s, CsChannels_s,])
        energies = np.hstack([BaEnergies, NaEnergies, CsEnergies,])
    
    '''
    Left out higher energy points due to possible quadratic mapping of observed energies.
    Could be due to saturation of detector. Explain this in the report, maybe show examples.
    '''
    
    energy_grid = np.linspace(min(energies), max(energies), num=100)
    
    # Linea Fit
    param, error = curve_fit(linear, energies, channels, 
                      p0 = [1, 1], sigma=channels_s)
    error = np.sqrt(np.diag(error))
    
    def f(x):
        return param[0] + param[1] * x
    
    if quad == True:
        # Quadratic Fit
        param_2, error_2 = curve_fit(quadratic, energies, channels, 
                                     p0=[1, 1, 1], sigma=channels_s)
        error_2 = np.sqrt(np.diag(error)).flatten()
        
        def f2(x):
            return param_2[0] + param_2[1] * x + param_2[2] * x**2
    
    
    chisq = np.sum(( (f(energies) - channels) / channels_s) ** 2)
    dof = len(channels) - len(param)
    rechisq = chisq/dof
    pval = chi2.sf(chisq, dof) * 100
    print('Conversion GOF Test:')
    print('Parameters:', param)
    print('Errors:', error)
    print('ChiSq:', chisq)
    print('dof:', dof)
    print('ChiSq/dof:', rechisq)
    print('p-value:', pval, '%')
    
    if quad == True:
        chisq2 = np.sum(( (f2(energies) - channels) / channels_s) ** 2)
        dof2 = len(channels) - len(param_2)
        rechisq2 = chisq2/dof2
        pval2 = chi2.sf(chisq2, dof2) * 100
        print('Parameters:', param_2)
        print('Errors:', error_2)
        print('ChiSq:', chisq2)
        print('dof:', dof2)
        print('ChiSq/dof:', rechisq2)
        print('p-value:', pval2, '%')
    
    fig, ax = plt.subplots()
    ax.errorbar(energies, channels, yerr = channels_s, fmt='.', color='black', label='Data')
    ax.plot(energy_grid, 
            linear(energy_grid, param[0], param[1]), '--', color='gray', label='Fit')
    ax.set_ylabel('Channel')
    ax.set_xlabel('Energy [KeV]')
    if quad == True:
        ax.plot(energy_grid, 
            quadratic(energy_grid, param_2[0], param_2[1], param_2[2]), '--', color='red')
        ax.yaxis.set_minor_locator(mtick.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mtick.MultipleLocator(20))
        ax.set_xlim(75, 1340)
        ax.set_ylim(40, 1060)
    else:
        ax.yaxis.set_minor_locator(mtick.MultipleLocator(10))
       # ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
        ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
        ax.set_xlim(75, 675)
        ax.set_ylim(50, 580)
    ax.legend()
    plt.tight_layout()
    #plt.close()
    '''
    def f(x):
        return (x - param[0]) / param[1]
    
    def f_s(f, x, dx):
        a = 
        return f*np.sqrt(((np.sqrt(error[0]**2 + dx**2) \ (x - param[0]))**2 + (error[1]/param[1])**2))
    '''
    def conversion(x, dx):
        a = x - param[0]
        da = np.sqrt(error[0]**2 + dx**2)
        b = param[1]
        db = error[1]
        f = a / b
        df = f * np.sqrt((da/a)**2 + (db/b)**2)
        return f, df
    
    return conversion

def compton():
    
    def linear(x, c0, c1):
        return c0 + c1 * x
    
    angles = np.array([30, 40, 50,60, 70, 80, 90, 100, 110, 180])
    angular = 1 - np.cos(np.deg2rad(angles))
    
    # old
    #channel = np.array([445.727, 420.641, 388.062,341.052,301.279,
    #                   268.953,242.199,213.918,199.539,140])
    channel = np.array([440.601, 415.826, 386.257, 341.406, 303.475, 271.29, 245.05, 215.67, 201.489, 162.401])
    channel_s = np.array([4.698, 3.485, 2.284, 2.374, 1.307, 1.276, 1.222, 1.384, 1.077, 7.056])
    
    chisqs = np.array([410.356, 424.845, 428.662, 470.778, 573.041, 457.722, 468.047, 282.07, 247.47, 957.131])
    dofs = np.array([750-11, 631-8, 657-8, 482-6, 654-6, 488-6, 437-6, 484-6, 312-6, 600-12])
    rechisqs = chisqs / dofs
    #pvals = [chi2.sf(chisqs[i], dofs[i]) for i in range(len(chisqs))]
    pvals = chi2.sf(chisqs, dofs) * 100
    
    ch2e = ch2energy()
    energies, energies_s = ch2e(channel, channel_s)
    invenergies = 1/energies
    invenergies_s = invenergies * energies_s / energies
    
    param, error = curve_fit(linear, angular, invenergies, 
                      p0 = [1, 1], sigma=invenergies_s)
    error = np.sqrt(np.diag(error))
    
    def f(x):
        return param[0] + param[1] * x
    
    chisq = np.sum(( (f(angular) - invenergies) / invenergies_s) ** 2)
    dof = len(invenergies) - len(param)
    rechisq = chisq/dof
    pval = chi2.sf(chisq, dof) * 100
    
    print('\nCompton GOF Test:')
    print('Parameters:', param)
    print('Errors:', error)
    print('ChiSq:', chisq)
    print('dof:', dof)
    print('ChiSq/dof:', rechisq)
    print('p-value:', pval, '%')
    
    energy = 1/param[0]
    energy_s = energy * error[0] / param[0]
    
    m_e = 1 / param[1] #/ (299792458)**2
    m_e_s = m_e * error[1] / param[1] #/ (299792458)**2
    
    print('\nResults:')
    print('137Cs Energy:', energy, '+/-', energy_s, 'KeV')
    print('Mass of electron:', m_e, '+/-', m_e_s, 'KeV/c^2')
    
    a = np.linspace(0, np.max(angular)+0.1, num=500)
    fig, ax = plt.subplots()
    ax.errorbar(angular, invenergies, yerr=invenergies_s, fmt='.', color='black', label='Data')
    ax.plot(a, f(a), '--', color='gray', label='Fit')
    ax.set_xlabel(r'$1-\cos(\theta)$')
    ax.set_ylabel("1/E' [1/KeV]")
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.0001))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.05))
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0.0016, 0.0054)
    ax.legend()
    plt.tight_layout()