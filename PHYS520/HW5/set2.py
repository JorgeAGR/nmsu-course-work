#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:27:25 2020

@author: jorgeagr
"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.tools.visualization import plot_histogram
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Settings for plots
golden_ratio = (np.sqrt(5) + 1) / 2
width = 25
height = width / golden_ratio

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 28
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 24

# Problem 2
def approx_H(theta=np.pi, N=10, plot=False):
    q = QuantumRegister(1)
    c = ClassicalRegister(1)
    
    error = {}
    for n in range(1, N+1):
    
        qc = QuantumCircuit(q,c)
        
        for j in range(n):
            qc.rx(theta/n,q[0])
            qc.rz(theta/n,q[0])
        
        qc.h(q[0])
        
        qc.measure(q,c)
        shots = 20000
        job = execute(qc, Aer.get_backend('qasm_simulator'),shots=shots)
        try:
            error[n] = (job.result().get_counts()['1']/shots)
        except:
            error[n] = 0
        
    if plot:
        hist = plot_histogram(error)
        return hist
    return error

def better_approx_H(theta=np.pi, N=10, plot=False):
    q = QuantumRegister(1)
    c = ClassicalRegister(1)
    
    error = {}
    for n in range(1, N+1):
    
        qc = QuantumCircuit(q,c)
        
        # New approximation
        for j in range(n):
            qc.rz(theta/2/n,q[0])
            qc.rx(theta/n,q[0])
            qc.rz(theta/2/n,q[0])
        
        qc.h(q[0])
        
        qc.measure(q,c)
        shots = 20000
        job = execute(qc, Aer.get_backend('qasm_simulator'),shots=shots)
        try:
            error[n] = (job.result().get_counts()['1']/shots)
        except:
            error[n] = 0
        
    if plot:
        hist = plot_histogram(error)
        return hist
    return error

N = 10
fraction = 16
step_size = np.pi/fraction
theta_grid = np.arange(0, 8*np.pi+step_size, step_size)
final_errors = np.array([approx_H(theta=theta, N=N)[N] for theta in theta_grid])

best_theta = theta_grid[np.argmin(final_errors)]
print('Optimal theta = {}/{}'.format(np.argmin(final_errors), fraction))

hist = approx_H(best_theta, plot=True)
hist.set_size_inches(width, height)
hist.tight_layout(pad=0.5)
hist.savefig('theta_hist.eps', dpi=200)

# Problem 3
better_final_errors = np.array([better_approx_H(theta=theta, N=N)[N] for theta in theta_grid])

better_best_theta = theta_grid[np.argmin(final_errors)]
print('Better approx optimal theta = {}/{}'.format(np.argmin(better_final_errors), fraction))

fig, ax = plt.subplots()
ax.plot(theta_grid/np.pi, final_errors, '-o', color='blue', label=r'$(R_x(\frac{\theta}{n}) R_z(\frac{\theta}{n}))^n$')
ax.plot(theta_grid/np.pi, better_final_errors, '-o', color='red', label=r'$(R_z(\frac{\theta}{2n})R_x(\frac{\theta}{n})R_z(\frac{\theta}{2n}))^n$')
ax.set_xlabel('Theta')
ax.set_ylabel('Error')
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g $\pi$'))
ax.xaxis.set_major_locator(mtick.MultipleLocator(base=1/2))
ax.xaxis.set_minor_locator(mtick.MultipleLocator(base=1/fraction))
ax.legend()
fig.tight_layout(pad=0.5)
fig.savefig('error_scale.eps', dpi=200)