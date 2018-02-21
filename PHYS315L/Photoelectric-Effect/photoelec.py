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

def avgVoltage(measurement, color1, color2, color3, color4, color5):
    v1 = []
    v2 = []
    v3 = []
    v4 = []
    v5 = []
    for m in measurement:
        v1.append(m[color1])
        v2.append(m[color2])
        v3.append(m[color3])
        v4.append(m[color4])
        v5.append(m[color5])
    c1 = ufloat(np.mean(v1), np.std(v1))
    c2 = ufloat(np.mean(v2), np.std(v2))
    c3 = ufloat(np.mean(v3), np.std(v3))
    c4 = ufloat(np.mean(v4), np.std(v4))
    c5 = ufloat(np.mean(v5), np.std(v5))
    return c1, c2, c3, c4, c5

def linearLSF(x, y, dy):
    def linear(x, a, b):
        return b * x + a
    
    popt, pcov = curve_fit(linear, x, y)
    
    fit = linear(x, *popt)
    
    chisq = np.sum(((y - fit)/dy)**2) #Chi Squared
    chire = chisq / (len(x) - len(popt)) # Chi-Reduced (Chi Squared / Dof)
    perr = np.sqrt(np.diag(pcov)) # Error in parameters a, b
    
    return fit, popt, perr, chisq, chire

# == Constants == #
e = 1.602 * 10 ** (-19) # C
c = 299792458 # m/s
wly = 578 # nm
wlg = 546.074
wlb = 435.835
wlv = 404.656
wluv = 365.483

# == Stopping Potential Volts for various colors == #
measurements = [
{'yellow': 0.829, 'green': 0.846, 'blue': 1.426, 'violet': 1.630, 'ultraviolet': 1.831}, # First Order Left
{'yellow': 0.711, 'green': 0.815, 'blue': 1.412, 'violet': 1.623, 'ultraviolet': 1.859}, # Second Order Left
{'yellow': 0.745, 'green': 0.859, 'blue': 1.345, 'violet': 1.680, 'ultraviolet': 1.974}, # First Order Right
{'yellow': 0.656, 'green': 0.740, 'blue': 1.431, 'violet': 1.630, 'ultraviolet': 1.903} # Second Order Right
] # Redo these numbers, they're wrong

volt_avg = avgVoltage(measurements, 'yellow', 'green', 'blue', 'violet', 'ultraviolet')
volt_avg_n = []
volt_avg_s = []

for i in range(len(volt_avg)):
    volt_avg_n.append(volt_avg[i].n)
    volt_avg_s.append(volt_avg[i].s)

freqs = c / ( np.array([wly, wlg, wlb, wlv, wluv]) * 10**(-9))

VFit, VPopt, VPerr, VChiSq, VChiRe = linearLSF(freqs, volt_avg_n, volt_avg_s)

# == Plots == #
fig1, ax1 = plt.subplots()
ax1.errorbar(freqs, volt_avg_n, yerr = volt_avg_s, fmt = 'o', label = 'Data')
ax1.plot(freqs, VFit, label = 'Fit')
text1 = 'Parameters: \n$h/e$ = %.3e \xb1 %.3e $J/A$\n\n $\chi^2/\\nu$ = %.3f' % (VPopt[1], VPerr[1], VChiRe)
ax1.text(6.5*10**14,0.8, text1, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})
ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3e'))
plt.xticks(rotation=45)
ax1.tick_params(axis='x',labelsize = 11)
ax1.grid()
ax1.set_title("Measurement of Planck's constant")
ax1.set_xlabel('Frequency $(Hz)$')
ax1.set_ylabel('Voltage $(V)$')
ax1.legend()


# == Intensisity Differences == #
measurementsYellow = {'20': 0.727,'40': 0.739,'60': 0.746,'80': 0.750, '100': 0.750}
measurementsBlue = {'20': 1.440, '40': 1.459, '60': 1.463, '80': 1.465, '100': 1.462}
