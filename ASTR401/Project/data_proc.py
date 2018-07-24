import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import astropy.coordinates as coord
import astropy.units as u

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 15
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [9.0, 9.0]

def linearLSF(x, y, dy):
    def linear(x, a, b):
        return b * x + a
    
    popt, pcov = curve_fit(linear, x, y)
    
    fit = linear(x, *popt)
    
    chisq = np.sum(((y - fit)/dy)**2) #Chi Squared
    chire = chisq / (len(x) - len(popt)) # Chi-Reduced (Chi Squared / Dof)
    perr = np.sqrt(np.diag(pcov)) # Error in parameters a, b
    
    return fit, popt, perr, chisq, chire

data = pd.read_csv('ddo_data.csv', sep = '|')

for key in data:
    for i in range(data.shape[0]):
        if (data.loc[i,key] == '###') or (data.loc[i,key] == '####'):
            data.loc[i,key] = np.nan

with open('gcvs_data.dat') as file:
    for line in file:
        if (line[40:44] == 'CEP ') and (line[62] == 'V') and not (line[77:90].isspace()):
            if line[55:60].isspace():
                data.loc[len(data)] = (line[0:7], line[8:12]+line[15:20], float(line[77:90]), float(line[48:53])/2, np.nan, np.nan)
            else:
                data.loc[len(data)] = (line[0:7], line[8:12]+line[15:20], float(line[77:90]), (float(line[48:53])+float(line[55:60]))/2, np.nan, np.nan)

data['ABS_MAG'] = -2.908 * np.log10(data['PERIOD']) - 1.203
data['DIST_APPROX'] = 10 ** ( (data['V_INTMEAN'] - data['ABS_MAG'] + 5) / 5 )

data.to_html('data.html')

periods = data['PERIOD'].values[~np.isnan(data['DIST_APPROX'].values)]
distances = data['DIST_APPROX'].values[~np.isnan(data['DIST_APPROX'].values)] * 1e-3 #kpc
logperiods = np.log10(data['PERIOD'].values[~np.isnan(data['DIST_APPROX'].values)])
m = data['V_INTMEAN'].values[~np.isnan(data['DIST_APPROX'].values)]
absmag = data['ABS_MAG'].values[~np.isnan(data['DIST_APPROX'].values)]
logdistances = np.log10(data['DIST_APPROX'].values[~np.isnan(data['DIST_APPROX'].values)])
glong = data['L'].values[~np.isnan(data['DIST_APPROX'].values)]
glat =  data['B'].values[~np.isnan(data['DIST_APPROX'].values)]

#mlogdfit, mlogdpopt, mlogdperr, mlogdchisq, mlogdchire = linearLSF(m, logdistances, np.std(logdistances))
#logplogdfit, logplogdpopt, logplogdperr, logplogdchisq, logplogdchire = linearLSF(logperiods, logdistances, np.std(logdistances))
logdmfit, logdmpopt, logdmperr, logdmchisq, logdmchire = linearLSF(logdistances, m, np.std(m))
'''
So logdmpopt[1] gives you the slope! Which should be related:
    m = 5log10(d) + ( M_abs - 5 )

Intercept logdmpopt[0] is roughly constant throughout the data, although not the same as the fit's value
'''

fig1, ax1 = plt.subplots() 
ax1.hist(logdistances, 6, color = 'black')
ax1.set_xlabel('log10(d)')
ax1.set_ylabel('Number of Cepheids')
fig1.suptitle('Cepheid Variables Distances', fontweight = 'bold')
'''
fig2, ax2 = plt.subplots()
ax2.plot(logdistances, m, '.')#, color = 'gray')
ax2.plot(logdistances, logdmfit)#, color = 'red')
ax2.set_xlabel('log10(d)')
ax2.set_ylabel('Apparent Magnitude')
fig2.suptitle('Distance vs Apparent Magnitude', fontweight = 'bold')
ax2.grid()

fig3, ax3 = plt.subplots()  
ax3.plot(logdistances, logperiods, '.') 
ax3.set_xlabel('log10(d)')
ax3.set_ylabel('log10(T)')
fig3.suptitle('Distance vs Pulsation Period', fontweight = 'bold')
ax3.grid()
'''

lat = coord.Angle(glat*u.degree)
long = coord.Angle(glong*u.degree)
'''
figpol1 = plt.figure()
figpol1.suptitle(r'Galactic Longitude vs Distance ($log_{10}d$)', fontweight = 'bold')
axpol1 = figpol1.add_subplot(111, projection = 'polar')
axlong = axpol1.scatter(long.radian, logdistances, marker = '.', c = logperiods, cmap = 'afmhot')#, cmap = 'YlOrBr')
axpol1.set_rmax(5)
colorlong = figpol1.colorbar(axlong)
colorlong.set_label(r'$log_{10}T$')

figpol2 = plt.figure()
figpol2.suptitle(r'Galactic Latitude vs Distance ($log_{10}d$)', fontweight = 'bold')
axpol2 = figpol2.add_subplot(111, projection = 'polar')
axlat = axpol2.scatter(lat.radian, logdistances, marker = '.', c = logperiods, cmap = 'afmhot')#, cmap = 'YlOrBr')
axpol2.set_thetamin(-20)
axpol2.set_thetamax(30)
axpol2.set_rmax(5)
colorlat = figpol2.colorbar(axlat)
colorlat.set_label(r'$log_{10}T$')

long = long.wrap_at(180*u.degree)
figmoll = plt.figure()
figmoll.suptitle('Galactic Coordinates: Mollweide Projection', fontweight = 'bold')
axmoll = figmoll.add_subplot(111, projection = 'mollweide')
axcoord = axmoll.scatter(long.radian, lat.radian, marker = '.', c = logdistances, cmap = 'YlOrBr')
axmoll.grid()
colorcoord = figmoll.colorbar(axcoord)
colorcoord.set_label(r'$log_{10}d$')
'''
# There is no limit...only if we can resolve Cepheids with telescopes or not
