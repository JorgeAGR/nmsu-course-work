import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [9.0, 9.0]

# == Constants == #

G = 6.676e-11 #w/e
m_sol = 1.989e30 #kg
kpc2m = 3.086e19 #m
m_star = 0.5 #solar mass

data = pd.read_csv('data.csv', sep = '|') # Reads in data

# == Manipulation == #

# Calculates half-light radius in meters
data['r_h (m)'] = (data['R_Sun'] * kpc2m) * np.tan(np.deg2rad(data['r_h'] / 60)) # m

# Calculates virial mass in kg
data['m_v'] = 5 * ( (data['sig_v'] * 1000) ** 2 ) * data['r_h (m)'] / G # kg

# Converts virial mass to solar masses
data['m_vsol'] = data['m_v'] /  m_sol # solar mass

# Calculates number of stars per cluster
data['n_stars'] = data['m_vsol'] / m_star # number

# == Histogram Formatting == #

masses = data['m_vsol'].tolist()
masses = np.asarray(masses)[~np.isnan(masses)]

fh, axh = plt.subplots()
axh.hist(np.log10(masses), color = 'gray')
axh.set_xlim(3, 7)

axh.set_xlabel(r'log$_{10}$(Virial Mass [$M_{\odot}$])')
axh.set_ylabel('Number of clusters')
fh.suptitle('Virial Mass of Clusters', fontweight = 'bold')
axh.grid()

# == Distance Plots Formatting == #

distances = data['R_gc'].tolist()
distances = np.asarray(distances)[~np.isnan(distances)]
metald = data['[Fe/H]'].tolist()
metald = np.asarray(metald)[~np.isnan(distances)]

f, ax = plt.subplots()
ax.plot(distances, metald, '.', color = 'gray')

ax.set_xlabel('Distance [kpc]')
ax.set_ylabel('[Fe/H]')
f.suptitle('Metalicity vs Distance from Galactic center', fontweight = 'bold')
ax.grid()
ax.set_xscale('log')

angles = data['B'].tolist()
angles = np.asarray(angles)[~np.isnan(angles)]
metala = data['[Fe/H]'].tolist()
metala = np.asarray(metala)[~np.isnan(angles)]

f2, ax2 = plt.subplots()
ax2.plot(angles, metala, '.', color = 'gray')

ax2.set_xlabel('Galactic Latitude [$^\circ$]')
ax2.set_ylabel('[Fe/H]')
f2.suptitle('Metalicity vs Galactic Latitude', fontweight = 'bold')
ax2.grid()

print(data.to_latex())