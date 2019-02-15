#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:05:53 2018

@author: jorgeagr
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

density = 2.7e3 # Al: mg/cm^3

thickness = np.array([0, 1.61, 3.41, 4.67, 7.10, 9.48, 13.2, 22.1, 29.3, 46.0, 67.7, 
             106, 133, 171, 221, 271, 300, 364, 444, 518]) # mg/cm^2

counts = np.array([7152, 6778, 6721, 6737, 6724, 6515, 6267, 6131, 5881, 5410, 4927,
          4477, 4000, 3524, 2981, 2265, 2066, 1371, 883, 600])
counts_err = np.sqrt(counts)

time = np.ones(len(counts)) * 30 #


time_interp = np.arange(30, 60*5+30, 30)
time_interp = [60*5, 90000]

time_fraccs = np.ones((len(time_interp), len(counts)-1))

for i, t in enumerate(time_interp):
    c = counts / time * t
    c_err = np.sqrt(c)
    
    differences = np.abs(np.diff(c))
    differences_err = np.array([np.sqrt(c_err[i]**2 + c_err[i+1]**2) for i in range(len(c)-1)])
    differences_fracc = differences_err / differences
    
    time_fraccs[i] = differences_fracc
    
fig, ax = plt.subplots()
ax.errorbar(thickness, counts, yerr=10*counts_err, fmt='.')
plt.close()

fig, ax = plt.subplots()
ax.axhline(0.1,linewidth=1, color='black')
for i, t in enumerate(time_interp):
    ax.plot(time_fraccs[i], '.', label = str(t) + ' s')
ax.legend()
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
plt.tight_layout()
plt.close()
# Have to taken LOOONGER measurements for 1.61 to 7.1 thicknesses

#============================================================================#

# Source: Sr-90
sr90_energy = 545.86e-3 # MeV
yt90_energy = 2.2801 # MeV
# Source: http://periodictable.com/Isotopes/038.90/index.dm.html
c = 299792458 # m/s
m_e = 9.10938356e-31 # kg
mc2 = 0.5109989461 # MeV

def E(r):
    a = 1.265
    b = -0.0954
    root = np.sqrt(a**2 - 4 * b *(-np.log(r) + np.log(412)))
    power = (- a + root) / (2*b)
    return np.exp(power)
    
def f(E):
    a = 1.265
    b = -0.0954
    return 412 * E ** (a - b * np.log(E))

def currie(E, C, E_0):
    W = E + mc2
    W_0 = E_0 + mc2
    p2 = W**2 - mc2**2
    return C * p2 * (W_0 - W)**2 #* fermi(E)

def currie_N(N, E):
    return np.sqrt(N / ( (E + mc2)**2 - mc2**2 ) / fermi(E) )

def currie_E(E, E_0):
    return E_0 - E

def fermi(E):
    W = E/mc2 + 1
    Z = 39 # Sr
    a = 5.5465e-3
    a_0 = 404.56e-3
    b = 76.929e-3
    b_0 = 73.184e-3
    A = 1 + a_0 * np.exp(b_0 * Z)
    B = a * Z * np.exp(b * Z)
    return np.sqrt(A + B / (W-1))

def linear(x, x0, x1, y0, y1):
    m = (x1 - x0) / (y1 - y0)
    return m*x + b

thickness = np.array([0, 1.61, 3.41, 4.67, 9.48, 13.2, 22.1, 29.3, 46.0, 67.7, 
             106, 133, 171, 221, 271, 300, 364, 444, 518]) # mg/cm^2
    
#energy = np.array([0, 2.5e-2, 4e-2, 4.5e-2, 7e-2, 8e-2, 1e-1, 1.25e-1, 1.5e-1, 
#                   2.5e-1, 2.5e-1, 3e-1, 4e-1, 4.5e-1, 6e-1, 6e-1, 7e-1, 8e-1, 9e-1])

counts = np.array([69945, 20616692, 20296109, 20133329, 65252, 63630, 60282, 58020, 
                   53906, 49651, 43922, 39662, 34861, 28520, 22859, 19980, 14267, 9095, 5955])
counts_err = np.sqrt(counts)

time = np.array([300, 90000, 90000, 90000, 300, 300, 300, 300,
                 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300])

counts = counts/time
counts_err = counts_err/time

distance = (thickness / density) * 10**(4)# cm -> mu m

# interpolate.interp1d(norm_time, lcf.flux, kind='cubic')

energy = E(thickness)
mid_energy = (energy[1:] + energy[:-1]) / 2
energy_width = np.abs(np.diff(energy))

freqs = np.abs(np.diff(counts))
freqs_err = np.sqrt(counts_err[:-1]**2 + counts_err[1:]**2)
freqs = freqs/energy_width
freqs_err = freqs_err/energy_width
freqs_normalized = freqs/np.sum(freqs)

fig, ax = plt.subplots()
ax.errorbar(distance, counts, yerr=10*counts_err, fmt='.')
ax.set_xlim(0-50, distance[-1]+50)
ax.set_xlabel(r'Distance [$\mu$m]')
ax.set_ylabel('Counts per second')
plt.tight_layout()
plt.close()

total_counts = np.sum(counts)
normalized_counts = counts/total_counts

e = np.linspace(0.01, max(energy))
e_sr = np.linspace(0.01, sr90_energy)
e_yt = np.linspace(0.01, yt90_energy)

sr_counts = currie(e_sr, 1, sr90_energy)
yt_counts = currie(e_yt, 1, yt90_energy)

fig, ax = plt.subplots()
ax.errorbar(mid_energy, freqs/max(freqs), xerr=energy_width/2, yerr=freqs_err/max(freqs), fmt='-', color='black')
plt.tight_layout()
plt.close()

#energy = energy[1:]
#counts = counts[1:]

fig, ax = plt.subplots()
ax.plot(energy, currie_N(normalized_counts, energy), '.')

#ax.plot(e_sr, currie_E(e_sr, sr90_energy))
#ax.plot(e_yt, currie_E(e_yt, yt90_energy))

ax.set_xlabel('Energy [MeV]')
ax.set_ylabel('N')
plt.tight_layout()

#plt.close()

#f_counts = interpolate.interp1d(np.arange(, counts/total_counts, kind='cubic')

'''
fig, ax = plt.subplots()
ax.errorbar(energy, counts, yerr=10*counts_err, fmt='.')
ax.set_xlim(0-0.01, energy[-1]+0.01)
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.1))
ax.set_xlabel('Energy [MeV]')
ax.set_ylabel('Counts per second')
    '''