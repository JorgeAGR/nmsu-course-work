# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 20:01:36 2018

@author: jorge
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit
from scipy.stats import chi2
from scipy import interpolate

golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

mc2 = 0.5109989461 # MeV
q =  1.6021766208e-19
E_sr90 = 545.86e-3
E_yt91 = 2.2801

file = np.loadtxt('estar.txt')
energies = file[:,0]
thickness = file[:,1]
f = interpolate.interp1d(thickness, energies, kind='cubic')

def E(r):
    a = 1.265
    b = -0.0954
    root = np.sqrt(a**2 - 4 * b *(-np.log(r) + np.log(412)))
    power = (- a + root) / (2*b)
    return np.exp(power)

def kurie(E, E_0):
    W = E + mc2
    W_0 = E_0 + mc2
    
    if E_0 == E_sr90:
        Z = 39
    else:
        Z = 40
    
    a = 5.5465e-3
    a_0 = 404.56e-3
    b = 76.929e-3
    b_0 = 73.184e-3
    A = 1 + a_0 * np.exp(b_0 * Z)
    B = a * Z * np.exp(b * Z)
    F = np.sqrt(A + B / (W-mc2))
    
    p2 = (W**2 - mc2**2)
    
    return (W_0 - W)**2 * p2 * F

def kurie_sr(E):
    E_0 = E_sr90
    W = E + mc2
    W_0 = E_0 + mc2
    Z = 39
    
    a = 5.5465e-3
    a_0 = 404.56e-3
    b = 76.929e-3
    b_0 = 73.184e-3
    A = 1 + a_0 * np.exp(b_0 * Z)
    B = a * Z * np.exp(b * Z)
    F = np.sqrt(A + B / (W-mc2))
    
    p2 = (W**2 - mc2**2)
    
    return (W_0 - W)**2 * p2 * F

def kurie_yt(E):
    E_0 = E_sr90
    W = E + mc2
    W_0 = E_0 + mc2
    Z = 40
    
    a = 5.5465e-3
    a_0 = 404.56e-3
    b = 76.929e-3
    b_0 = 73.184e-3
    A = 1 + a_0 * np.exp(b_0 * Z)
    B = a * Z * np.exp(b * Z)
    F = np.sqrt(A + B / (W-mc2))
    
    p2 = (W**2 - mc2**2)
    
    return (W_0 - W)**2 * p2 * F

def kurie_dE(N, E):
    W = E + mc2
    Z = 39
    
    a = 5.5465e-3
    a_0 = 404.56e-3
    b = 76.929e-3
    b_0 = 73.184e-3
    A = 1 + a_0 * np.exp(b_0 * Z)
    B = a * Z * np.exp(b * Z)
    F = np.sqrt(A + B / (W-mc2))
    
    return np.sqrt(N / (W**2 - mc2**2) / F)

def kurie_recover(N, E):
    W = E + mc2
    Z = 39
    
    a = 5.5465e-3
    a_0 = 404.56e-3
    b = 76.929e-3
    b_0 = 73.184e-3
    A = 1 + a_0 * np.exp(b_0 * Z)
    B = a * Z * np.exp(b * Z)
    F = np.sqrt(A + B / (W-mc2))
    
    p2 = (W**2 - mc2**2)
    
    return N**2 * p2 * F

def linear(x, c_0, c_1):
    return c_0 + c_1 * x

def gauss(a, b, n, f):
    x, w = np.polynomial.legendre.leggauss(n)
    i = ((b - a) / 2 ) * np.sum(f(((b-a)*x/2 + (b+a)/2)) * w )
    return i

density = 2.7e3 # Al: mg/cm^3

thickness = np.array([0, 1.61, 3.41, 4.67, 9.48, 13.2, 22.1, 29.3, 46.0, 67.7, 
             106, 133, 171, 221, 271, 300, 364, 444, 518]) # mg/cm^2


#=============================================================================#
def time_recording():
    thickness = np.array([0, 1.61, 3.41, 4.67, 7.10, 9.48, 13.2, 22.1, 29.3, 46.0, 67.7, 
             106, 133, 171, 221, 271, 300, 364, 444, 518])
    
    counts = np.array([7152, 6778, 6721, 6737, 6724, 6515, 6267, 6131, 5881, 5410, 4927,
              4477, 4000, 3524, 2981, 2265, 2066, 1371, 883, 600])
    counts_err = np.sqrt(counts)
    
    time = np.ones(len(counts)) * 30 #
    
    time_interp = np.arange(30, 60*5+30, 30)
    time_interp = [30, 60*5, 90000]
    colors = ['black', 'red', 'gray']
    
    time_fraccs = np.ones((len(time_interp), len(counts)-1))
    
    for i, t in enumerate(time_interp):
        c = counts / time * t
        c_err = np.sqrt(c)
        
        differences = np.abs(np.diff(c))
        differences_err = np.array([np.sqrt(c_err[i]**2 + c_err[i+1]**2) for i in range(len(c)-1)])
        differences_fracc = differences_err / differences
        
        time_fraccs[i] = differences_fracc * 100
        
    fig, ax = plt.subplots()
    ax.errorbar(thickness, counts/time, yerr=10*counts_err/time, fmt='.', color='black')
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(10))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(50))
    ax.xaxis.set_major_locator(mtick.MultipleLocator(50))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
    ax.set_xlabel(r'Distance [mg/cm$^2$]')
    ax.set_ylabel('Counts [1/s]')
    plt.tight_layout()
    
    superbad = np.array([1, 2, 3])
    fig, ax = plt.subplots()
    ax.axhline(10,linewidth=1, color='gray', linestyle='--')
    for i, t in enumerate(time_interp):
        ax.plot(time_fraccs[i], '.', label = str(t) + ' s', color=colors[i])
        if i == 1:
            break
    ax.plot(superbad, time_fraccs[2][superbad], '.', color='orange', label='90000')
    ax.legend()
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(20))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(100))
    ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
    ax.set_xlabel(r'Index i ($d_{i+1} - d_i$)')
    ax.set_ylabel('Fractional Uncertainty [%]')
    plt.tight_layout()
    #plt.close()

#=============================================================================#
#time_recording()

counts = np.array([69945, 20616692, 20296109, 20133329, 65252, 63630, 60282, 58020, 
                   53906, 49651, 43922, 39662, 34861, 28520, 22859, 19980, 14267, 9095, 5955])
counts_err = np.sqrt(counts)

time = np.array([300, 90000, 90000, 90000, 300, 300, 300, 300,
                 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300])

counts = counts/time
counts_err = counts_err/time

fig, ax = plt.subplots()
ax.errorbar(thickness, counts, yerr=10*counts_err, fmt='.', color='black')
ax.yaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.yaxis.set_major_locator(mtick.MultipleLocator(50))
ax.xaxis.set_major_locator(mtick.MultipleLocator(50))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
ax.set_xlabel(r'Distance [mg/cm$^2$]')
ax.set_ylabel('Counts [1/s]')
plt.tight_layout()
#plt.close()

#=============================================================================#

distance = (thickness / density) * 10**(4)# cm -> mu m

energy = E(thickness)
mid_energy = (energy[1:] + energy[:-1]) / 2
energy_width = np.abs(np.diff(energy))

freqs = np.abs(np.diff(counts))
freqs_err = np.sqrt(counts_err[:-1]**2 + counts_err[1:]**2)
freqs = freqs/energy_width
freqs_err = freqs_err/energy_width
freqs_normalized = freqs/np.sum(freqs)

freqs_doubled = np.zeros(len(freqs) * 2 + 1)
for i, f in enumerate(freqs):
    freqs_doubled[2*i] = f
    freqs_doubled[2*i + 1] = f

energy_doubled = np.zeros(len(freqs) * 2 + 1)
i = 1
for e in energy[1:]:
    energy_doubled[i] = e
    i += 1
    if i < 37:
        energy_doubled[i] = e
        i += 1

#e_sr90 = np.linspace(0.01, E_sr90, num=500)
#e_yt91 = np.linspace(0, E_yt91, num=500)
e_sr90 = energy[1:6]
e_yt91 = energy[-5:]

#counts_sr90 = kurie(e_sr90, E_sr90)
#counts_yt91 = kurie(e_yt91, E_yt91)
counts_sr90 = counts[1:6]
counts_yt91 = counts[-5:]

#=============================================================================#

fig, ax = plt.subplots()
ax.errorbar(mid_energy, freqs, xerr=energy_width/2, 
            yerr=freqs_err, fmt='.', color='black')
ax.plot(energy_doubled, freqs_doubled, color='black')
ax.plot(mid_energy, freqs, '--', color='gray')
ax.set_xlim(0, 1.22)
ax.set_ylim(50, 400)
ax.set_xlabel('Energy [MeV]')
ax.set_ylabel('Counts [1/s]')
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.02))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(10))
plt.tight_layout()
#plt.close()

#=============================================================================#
e_grid_sr = np.linspace(0.01, E_sr90, num=200)
e_grid_yt = np.linspace(0.01, E_yt91, num=200)
fig, ax = plt.subplots()
ax.errorbar(mid_energy, freqs/max(freqs), xerr=energy_width/2, 
            yerr=freqs_err/max(freqs), fmt='.', color='black')
ax.plot(energy_doubled, freqs_doubled/max(freqs), color='black',
        label='Data')

# Theory
c_sr = kurie(e_grid_sr, E_sr90) / gauss(0, E_sr90, 10, kurie_sr)
c_sr = c_sr/max(c_sr)
#ax.plot(e_grid_sr, c_sr)
c_yt = kurie(e_grid_yt, E_yt91) / gauss(0, E_yt91, 10, kurie_yt)
c_yt = c_yt/max(c_yt)
#ax.plot(e_grid_yt, c_yt)

delta = 0.02
e_sr = np.arange(0.00000000001, E_sr90+delta, delta)
c_sr = kurie(e_sr, E_sr90) / gauss(0, E_sr90, 10, kurie_sr)
i_sr = kurie_dE(c_sr, e_sr)
i_sr = i_sr / (E_sr90 * max(i_sr) / 2)
norm = max(i_sr)
i_sr = i_sr / norm
i_sr = kurie_recover(i_sr, e_sr)

e_yt = np.arange(0.00000000001, E_yt91+delta, delta)
c_yt = kurie(e_yt, E_yt91) / gauss(0, E_yt91, 10, kurie_yt)
i_yt = kurie_dE(c_yt, e_yt)
i_yt = i_yt / (E_yt91 * max(i_yt) / 2)
i_yt = i_yt / norm
i_yt = kurie_recover(i_yt, e_yt)

i_tot = i_sr+i_yt[:len(i_sr)]
i_tot = np.hstack([i_tot, i_yt[len(i_sr):]])
peak = max(i_tot)
i_sr = i_sr / peak
i_yt = i_yt / peak
i_tot = i_tot / peak

ax.plot(e_sr, i_sr, '--', color='blue', label=r'$^{90}$Sr')
ax.plot(e_yt, i_yt, '--', color='green', label=r'$^{90}$Y')
ax.plot(e_yt, i_tot, color='red', label='Total')

ax.set_xlim(0, 1.22)
ax.set_ylim(0, 1.25)
ax.set_xlabel('Energy [MeV]')
ax.set_ylabel('Normalized Counts')
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.02))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.05))
ax.legend()
plt.tight_layout()
#plt.close()
#=============================================================================#
I_all = kurie_dE(freqs, mid_energy)
I_all_err = I_all * freqs_err / freqs

e_sr = mid_energy[:9]
e_yt = mid_energy[9:]
#I_sr = kurie_dE(freqs[:9], e_sr)
#I_err_sr = I_sr * freqs_err[:9] / freqs[:9]
#I_yt = kurie_dE(freqs[9:], e_yt)
#I_err_yt = I_yt * freqs_err[9:] / freqs[9:]
I_sr = I_all[:9]/I_all[0]
I_err_sr = I_all_err[:9]/I_all[0]
I_yt = I_all[9:]/I_all[0]
I_err_yt = I_all_err[9:]/I_all[0]

param_sr, error_sr = curve_fit(linear, e_sr, I_sr, sigma=I_err_sr)
error_sr = np.sqrt(np.diag(error_sr))
param_yt, error_yt = curve_fit(linear, e_yt, I_yt, sigma=I_err_yt)
error_yt = np.sqrt(np.diag(error_yt))

e_grid_sr = np.linspace(0.00000000001, -param_sr[0]/param_sr[1], num=200)
e_grid_yt = np.linspace(0.00000000001, -param_yt[0]/param_yt[1], num=200)
i_grid_sr = linear(e_grid_sr, param_sr[0], param_sr[1])
i_grid_yt = linear(e_grid_yt, param_yt[0], param_yt[1])

e_sr_found = -param_sr[0]/param_sr[1]
e_sr_found_err = e_sr_found * np.sqrt((param_sr[0]/error_sr[0])**2 + (param_sr[1]/error_sr[1])**2)
e_yt_found = -param_yt[0]/param_yt[1]
e_yt_found_err = e_yt_found * np.sqrt((param_yt[0]/error_yt[0])**2 + (param_yt[1]/error_yt[1])**2)

#=============================================================================#

fig, ax = plt.subplots()
ax.errorbar(mid_energy, I_all/I_all[0], yerr=I_all_err/I_all[0], 
            xerr=energy_width/2, fmt='.', color='black', label='Data')
ax.set_xlim(0, 1.22)
ax.set_ylim(0)
ax.set_xlabel('Energy [MeV]')
ax.set_ylabel('Intensity')
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.02))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.legend()
plt.tight_layout()
#plt.close()

#=============================================================================#

print('\nFit Results')
print('Sr90 E_0:', e_sr_found, '+/-', e_sr_found_err)
chisq = np.sum(( ( linear(e_sr, param_sr[0], param_sr[1]) - I_sr) / I_err_sr ) ** 2)
dof = len(e_sr) - len(param_sr)
rechisq = chisq/dof
pval = chi2.sf(chisq, dof) * 100
print('Fit Sr90 GOF Test:')
print('Parameters:', param_sr)
print('Errors:', error_sr)
print('ChiSq:', chisq)
print('dof:', dof)
print('ChiSq/dof:', rechisq)
print('p-value:', pval, '%')

print('\nYt91 E_0:', e_yt_found, '+/-', e_yt_found_err)
chisq = np.sum(( ( linear(e_yt, param_yt[0], param_yt[1]) - I_yt) / I_err_yt ) ** 2)
dof = len(e_yt) - len(param_yt)
rechisq = chisq/dof
pval = chi2.sf(chisq, dof) * 100
print('Fit Y90 GOF Test:')
print('Parameters:', param_yt)
print('Errors:', error_yt)
print('ChiSq:', chisq)
print('dof:', dof)
print('ChiSq/dof:', rechisq)
print('p-value:', pval, '%')

fig, ax = plt.subplots()
#ax.errorbar(mid_energy, kurie_dE(freqs, mid_energy), fmt='.')

#ax.errorbar(mid_energy, I_all/I_all[0], yerr=I_all_err/I_all[0], xerr=energy_width/2, fmt='.', color='black')
#ax.plot(e_grid_sr, i_grid_sr/np.max(i_grid_sr), '--', color='gray')
#ax.plot(e_grid_yt, i_grid_yt/np.max(i_grid_sr), '--', color='gray')

ax.errorbar(mid_energy, I_all/I_all[0], yerr=I_all_err/I_all[0], xerr=energy_width/2, 
            fmt='.', color='black', label='Data')
ax.plot(e_grid_sr, i_grid_sr, '--', color='gray', label='Fit')
ax.plot(e_grid_yt, i_grid_yt, '--', color='gray')

ax.set_xlim(0, 1.54)
ax.set_ylim(0)
ax.set_xlabel('Energy [MeV]')
ax.set_ylabel('Intensity')
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.02))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.legend()
plt.tight_layout()
#plt.close()

#=============================================================================#

fig, ax = plt.subplots()
#ax.errorbar(mid_energy, kurie_dE(freqs, mid_energy), fmt='.')

ax.errorbar(mid_energy, I_all/I_all[0], yerr=I_all_err/I_all[0], xerr=energy_width/2, 
            fmt='.', color='black', label='Data')

# THEORY
e_sr_lin = np.linspace(0.00000000001, E_sr90, num=200)
c_sr = kurie(e_sr_lin, E_sr90) / gauss(0, E_sr90, 10, kurie_sr)
i_sr = kurie_dE(c_sr, e_sr_lin)
area_sr = (E_sr90 * max(i_sr) / 2)
i_sr = i_sr / area_sr
norm = max(i_sr)
i_sr = i_sr / norm
ax.plot(e_sr_lin, i_sr, color='red', label='Theory')

c_sr_points = kurie(e_sr, E_sr90) / gauss(0, E_sr90, 10, kurie_sr)
i_sr_points = kurie_dE(c_sr_points, e_sr)
i_sr_points = i_sr_points / area_sr
i_sr_points = i_sr_points / norm

e_yt_lin = np.linspace(0.00000000001, E_yt91, num=200)
c_yt = kurie(e_yt_lin, E_yt91) / gauss(0, E_yt91, 10, kurie_yt)
i_yt = kurie_dE(c_yt, e_yt_lin)
area_yt = (E_yt91 * max(i_yt) / 2)
i_yt = i_yt / area_yt
i_yt = i_yt / norm
ax.plot(e_yt_lin, i_yt, color='red')

c_yt_points = kurie(e_yt, E_yt91) / gauss(0, E_yt91, 10, kurie_yt)
i_yt_points = kurie_dE(c_yt_points, e_yt)
i_yt_points = i_yt_points / area_yt
i_yt_points = i_yt_points / norm
print('--------------------')
print('\nThoery Results')
chisq = np.sum(( ( i_sr_points - I_sr) / I_err_sr ) ** 2)
dof = len(e_sr) - len(param_sr)
rechisq = chisq/dof
pval = chi2.sf(chisq, dof) * 100
print('Theory Sr90 GOF Test:')
print('Parameters:', param_sr)
print('Errors:', error_sr)
print('ChiSq:', chisq)
print('dof:', dof)
print('ChiSq/dof:', rechisq)
print('p-value:', pval, '%')

chisq = np.sum(( ( i_yt_points - I_yt) / I_err_yt ) ** 2)
dof = len(e_yt) - len(param_yt)
rechisq = chisq/dof
pval = chi2.sf(chisq, dof) * 100
print('Fit Y90 GOF Test:')
print('Parameters:', param_yt)
print('Errors:', error_yt)
print('ChiSq:', chisq)
print('dof:', dof)
print('ChiSq/dof:', rechisq)
print('p-value:', pval, '%')

#ax.set_xlim(0, 1.54)
ax.set_xlim(0, E_yt91)
ax.set_ylim(0)
ax.set_xlabel('Energy [MeV]')
ax.set_ylabel('Intensity')
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.02))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
ax.legend()
plt.tight_layout()
#plt.close()

#=============================================================================#