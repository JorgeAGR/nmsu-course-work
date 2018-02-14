import numpy as np
from uncertainties import ufloat, unumpy
import uncertainties.umath as umath
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick

mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [9.0, 9.0]

def shiftTheta(measurement, center):
    shift = measurement[center]
    for i in measurement:
        measurement[i] = np.abs(measurement[i] - shift)

def avgTheta(measurement, color1, color2, color3, color4, color5):
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []
    std = 0.1
    for m in measurement:
        t1.append(m[color1])
        t2.append(m[color2])
        t3.append(m[color3])
        t4.append(m[color4])
        if color5:
            t5.append(m[color5])
    c1 = ufloat(np.mean(t1), np.std(t1))
    c2 = ufloat(np.mean(t2), np.std(t2))
    c3 = ufloat(np.mean(t3), np.std(t3))
    c4 = ufloat(np.mean(t4), np.std(t4))
    '''c1 = ufloat(np.mean(t1), std)
    c2 = ufloat(np.mean(t2), std)
    c3 = ufloat(np.mean(t3), std)
    c4 = ufloat(np.mean(t4), std)'''
    if color5:
        c5 = ufloat(np.mean(t5), np.std(t5))
        return c1, c2, c3, c4, c5
    return c1, c2, c3, c4

def wavelength(d, theta, m):
    t = umath.radians(theta)
    sint = umath.sin(t)
    l = d * sint / m
    return l, sint

def linearLSF(x, y, dy):
    def linear(x, a, b):
        return b * x + a
    
    popt, pcov = curve_fit(linear, x, y)
    
    fit = linear(x, *popt)
    
    chisq = np.sum(((y - fit)/dy)**2) #Chi Squared
    chire = chisq / (len(x) - len(popt)) # Chi-Reduced (Chi Squared / Dof)
    perr = np.sqrt(np.diag(pcov)) # Error in parameters a, b
    
    return fit, popt, perr, chisq, chire

d = 1/(600 / (10**(-3))) # "Slit Size" m

# == Mercury == #
mHg = 1

measurementsHg = [
{'blue': 139.4, 'violet': 124.4, 'green': 120.2, 'yellow': 119.1, 'orange': 118.9}, # Angle Patrick Right
{'blue': 139.4, 'violet': 154.5, 'green': 158.7, 'yellow': 159.7, 'orange': 159.9}, # Angle Patrick Left
{'blue': 139.3, 'violet': 124.1, 'green': 120.0, 'yellow': 119.0, 'orange': 118.9}, # Angle Jorge Right
{'blue': 139.3, 'violet': 154.5, 'green': 158.4, 'yellow': 159.5, 'orange': 159.6}, # Angle Jorge Left
{'blue': 139.4, 'violet': 124.3, 'green': 120.1, 'yellow': 119.0, 'orange': 118.9}, # Angle Ryan Right
{'blue': 139.4, 'violet': 154.5, 'green': 158.5, 'yellow': 159.6, 'orange': 159.7}, # Angle Ryan Left
]

for m in measurementsHg:
    shiftTheta(m, 'blue')

theta_vHg_avg, theta_gHg_avg, theta_yHg_avg, theta_oHg_avg = avgTheta(measurementsHg, 'violet', 'green', 'yellow', 'orange', False)

vHg, sintvHg = wavelength(d, theta_vHg_avg, 1)
gHg, sintgHg = wavelength(d, theta_gHg_avg, 1)
yHg, sintyHg = wavelength(d, theta_yHg_avg, 1)
oHg, sintoHg = wavelength(d, theta_oHg_avg, 1)

sintheta = np.asarray([sintvHg.n, sintgHg.n, sintyHg.n, sintoHg.n])
sintheta_ds = np.asarray([sintvHg.s, sintgHg.s, sintyHg.s, sintoHg.s])
mlambda = np.asarray([vHg.n, gHg.n, yHg.n, oHg.n])
mlambda_ds = np.asarray([vHg.s, gHg.s, yHg.s, oHg.s])
dFit, dPopt, dPerr, dChiSq, dChiRe = linearLSF(sintheta, mlambda, mlambda_ds)

figHg, axHg = plt.subplots()
axHg.errorbar(sintheta, mlambda, xerr = sintheta_ds, yerr = mlambda_ds, fmt = 'o', label = 'Data Hg')
axHg.plot(sintheta, dFit, label = 'Fit')
axHg.set_title('Mercury (Hg)')
axHg.set_xlabel('$sin\\theta$')
axHg.set_ylabel('m$\lambda$ $(m)$')
textHg = 'Parameters:\nd = %.1e \xb1 %.1e m\n\n $\chi^2/\\nu$ = %.1e' % (dPopt[1], dPerr[1], dChiRe)
axHg.text(0.32,4.75*10**(-7), textHg, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})
axHg.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.yticks(rotation=45)
axHg.tick_params(axis='y',labelsize = 11)
axHg.grid()
axHg.legend()

# == Hydrogren == #
mH = 1

measurementsH = [
{'pink': 139.5, 'violet': 124.2, 'cyan': 122.2, 'red': 116.0, 'blue': 103.4}, # Angle Patrick Right
{'pink': 139.5, 'violet': 154.4, 'cyan': 156.2, 'red': 162.5, 'blue': 174.9}, # Angle Patrick Left
{'pink': 139.3, 'violet': 124.1, 'cyan': 122.3, 'red': 116.0, 'blue': 103.3}, # Angle Jorge Right
{'pink': 139.3, 'violet': 154.4, 'cyan': 156.3, 'red': 162.5, 'blue': 174.8}, # Angle Jorge Left
{'pink': 139.3, 'violet': 124.2, 'cyan': 122.3, 'red': 116.0, 'blue': 103.4}, # Angle Ryan Right
{'pink': 139.3, 'violet': 154.5, 'cyan': 156.3, 'red': 162.5, 'blue': 174.9}, # Angle Ryan Left
]

for m in measurementsH:
    shiftTheta(m, 'pink')

theta_vH_avg, theta_cH_avg, theta_rH_avg, theta_bH_avg = avgTheta(measurementsH, 'violet', 'cyan', 'red', 'blue', False)

vH, sintvH = wavelength(d, theta_vH_avg, 1)
cH, sintcH = wavelength(d, theta_cH_avg, 1)
rH, sintrH = wavelength(d, theta_rH_avg, 1)
bH, sintbH = wavelength(d, theta_bH_avg, 2)

lambda_inverse = 1 / np.array([vH, cH, rH])
l_i = unumpy.nominal_values(lambda_inverse)
l_i_ds = unumpy.std_devs(lambda_inverse)
ni_inverse_sq = 1 / (np.array([3, 4, 5]))**2
rFit, rPopt, rPerr, rChiSq, rChiRe = linearLSF(ni_inverse_sq, l_i, l_i_ds)

figH, axH = plt.subplots()
axH.errorbar(ni_inverse_sq, l_i, yerr = l_i_ds, fmt = 'o', label = 'H Data')
axH.plot(ni_inverse_sq, rFit, label = 'Fit')
textH = 'Parameters: \n$(R/n_f)^2$ = %.1f \xb1 %.1f\nR = %.1f \xb1 %.1f $m^{-1}$\n\n $\chi^2/\\nu$ = %.3f' % (rPopt[0], rPerr[0], rPopt[1], rPerr[1], rChiRe)
axH.text(0.08,1600000, textH, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})
axH.set_title('Hydrogen (H)')
axH.set_ylabel('$1/\lambda$ $(m^{-1})$')
axH.set_xlabel('$1/n_i^2$')
axH.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.yticks(rotation=45)
axH.tick_params(axis='y',labelsize = 11)
axH.grid()
axH.legend()

# == Unknown A == #
measurementsA = [
{'center': 139.4 , 'color1': 124.8, 'color2': 124.7, 'color3': 123.9, 'color4': 119.6, 'color5': 111.0}, # Angle Patrick Right
{'center': 139.4, 'color1': 153.7, 'color2': 153.6, 'color3': 154.9, 'color4': 155.0, 'color5': 157.3}, # Angle Ryan Left
]

for m in measurementsA:
    shiftTheta(m, 'center')

theta_c1A_avg, theta_c2A_avg, theta_c3A_avg, theta_c4A_avg, theta_c5A_avg = avgTheta(measurementsA, 'color1', 'color2', 'color3', 'color4', 'color5')

c1A, sintc1A = wavelength(d, theta_c1A_avg, 1)
c2A, sintc2A = wavelength(d, theta_c2A_avg, 1)
c3A, sintc3A = wavelength(d, theta_c3A_avg, 1)
c4A, sintc4A = wavelength(d, theta_c4A_avg, 1)
c5A, sintc5A = wavelength(d, theta_c5A_avg, 1)