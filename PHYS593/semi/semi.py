import os
import numpy as np
import scipy as sp
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

pathdir = 'data/'

epsilon = 1.054e-12 # F/cm
q =  1.6021766208e-19 # C
density_si = 2.3290 # g/cm^3
density_al = 2.7 # g/cm^3
d_depleted = 465e-4 # cm

E_0 = 5.486 # MeV, alpha energy

def clean_files(): # Already clean!
    files = os.listdir(pathdir)
    for f in files:
        with open(pathdir + f) as file:
            text = file.readlines()
            title = pathdir + f.rstrip('.Spe') + '_raw.npy'
            print('Deleting heads and tails:', f)
            del text[-14:]
            del text[0:12]
            with open(pathdir + f.rstrip('.Spe') + '_raw.txt', 'w+') as file:
                for num in text:
                    print(num, file=file)
            for i in range(len(text)):
                text[i] = int(text[i].rstrip('\n').lstrip(' '))
            text = np.array(text)
            np.save(title, text)


def setup():
    def calibration_func(x, A, k, C):
        
        return A * (1 - np.exp(-k*x)) + C
    
    voltage_supply = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]) # V
    
    total_resistance = 101 #M Ohm
    voltage_drop = np.array([0.01, 0.041, 0.084, 0.114, 0.135, 0.149, 0.157, 0.173, 0.184, 0.190, 0.198]) * total_resistance # V
    # 100 M Ohm from the preamp
    # 1 M Ohm from the resistor within the bias supply
    
    voltage_bias = voltage_supply - voltage_drop
    
    #current = 
    
    test = np.load(pathdir + 'am241_0V_raw.npy')
    
    '''
    # Fits w/o constant background (except last one)
    channels = np.array([1012.97, 1055.55, 1143.05, 1433.26, 1502.61, 1513.8, 1532.99, 1529.75, 1538.33, 1540.63, 1541.27])
    channels_s = np.array([5.01072, 5.12475, 5.67346, 3.06534, 2.00336, 2.70789, 1.90187, 1.84119, 1.60773, 2.41848, 1.66222])
    chisqs = np.array([485.492, 510.336, 519.729, 321.131, 383.774, 605.961, 724.148, 755.221, 360.203, 538.019, 1454.28])
    dofs = np.array([478, 510-4, 516-4, 331-4, 393-4, 609-4, 718, 735, 260, 490, 1306])
    '''
    channels = np.array([1007.13, 1052.5, 1134.39, 1432.69, 1501.5, 1512.53, 1530.19, 1528.54, 1537.61, 1540.22, 1541.27])
    channels_s = np.array([4.92662, 4.84387, 5.18497, 2.74768, 1.82336, 2.14455, 1.76431, 1.78499, 1.57198, 1.91554, 1.66222])
    chisqs = np.array([1049.63, 1115.12, 1075.51, 989.195, 1038.66, 738.894, 1064.08, 1210.34, 1570.53, 944.086, 1454.28])
    dofs = np.array([1021, 1113, 1195, 1217, 1305, 1180, 1229, 1260, 1243, 1331, 1306])
    rechisqs = chisqs / dofs
    pvals = chi2.sf(chisqs, dofs) * 100
    
    param, error = curve_fit(calibration_func, voltage_bias, channels, p0=[1, 1, 1])
    error = np.sqrt(np.diag(error))
    
    v = np.linspace(0, 50, num=200)
    fig, ax = plt.subplots()
    ax.errorbar(voltage_bias, channels, yerr=channels_s, fmt='.', color='black')
    #ax.plot(v, calibration_func(v, param[0], param[1], param[2]), '--', color='gray')
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('Peak Channel')
    ax.set_ylim(1020, 1560)
    ax.set_xlim(-0.5, 31)
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(20))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(100))
    plt.tight_layout()

def calibrate():
    def linear(x, c_0, c_1):
        return c_0 + c_1 * x
    
    channels = np.array([281.293, 559.814, 838.367, 1117.11, 1397.28, 1531.61, 1678.16, 1957.95])
    channels_s = np.array([0.0991482, 0.16209, 0.166326, 0.151908, 0.187174, 0.113161, 0.145929, 0.12916])
    chisqs = np.array([40.5822, 51.8591, 45.9096, 46.7066, 73.9286, 34.9002, 52.8336, 47.552])
    dofs = np.array([74, 59, 56, 67, 68, 76, 72, 83])
    rechisqs = chisqs / dofs
    pvals = chi2.sf(chisqs, dofs) * 100
    
    energies = np.array([1, 2, 3, 4, 5, 5.486, 6, 7]) # MeV
    
    param, error = curve_fit(linear, energies, channels, sigma=channels_s)
    error = np.sqrt(np.diag(error))
    
    def f(energy):
        return param[0] + param[1] * energy
    
    chisq = np.sum(( (f(energies) - channels) / channels_s) ** 2)
    dof = len(channels) - len(param)
    rechisq = chisq/dof
    pval = chi2.sf(chisq, dof) * 100
    
    print('\nCalibration GOF Test:')
    print('Parameters:', param)
    print('Errors:', error)
    print('ChiSq:', chisq)
    print('dof:', dof)
    print('ChiSq/dof:', rechisq)
    print('p-value:', pval, '%')
    
    e = np.linspace(0, 8, num=200)
    fig, ax = plt.subplots()
    ax.errorbar(energies, channels, yerr=channels_s, fmt='.', color='black')
    ax.plot(e, f(e), '--', color='gray')
    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel('Channel')
    plt.tight_layout()
    ax.set_ylim(250, 2000)
    ax.set_xlim(0, 8)
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(50))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(250))
    plt.close()
    
    def conversion(x, dx):
        a = x - param[0]
        da = np.sqrt(error[0]**2 + dx**2)
        b = param[1]
        db = error[1]
        f = a / b
        df = f * np.sqrt((da/a)**2 + (db/b)**2)
        return f, df
    
    return conversion

def energy2range():
    file = np.loadtxt('astar.txt')
    e, r = file[:, 0], file[:, 1]
    f = sp.interpolate.interp1d(e, r, kind='cubic')
    return f

def examples():
    #eg_hist = [np.load('data/am241_0V_raw.npy'), np.load('data/am241_10V_raw.npy')]
    #eg_titles = ['0', '10']
    eg_hist = [np.load('data/am241_50V_raw.npy'), np.load('data/energy_calibration_raw.npy')]
    eg_titles = ['50', 'N/A']
    eg_channel = np.arange(1, len(eg_hist[0])+1)
    
    fig, axis = plt.subplots(nrows=2, sharex=True)
    for i, ax in enumerate(axis):
        title = 'Supply Voltage: ' + eg_titles[i]+' V'
        ax.set_title(title)
        ax.plot(eg_channel, eg_hist[i], 'o', color='black', label='Data')
        ax.errorbar(eg_channel, eg_hist[i], yerr=np.sqrt(eg_hist[i]), fmt=',', color='gray', label='Error')
        ax.set_xlim(0, max(eg_channel))
        ax.set_ylim(0, max(eg_hist[i]) + 6)
        ax.set_ylabel('Counts')
        ax.xaxis.set_minor_locator(mtick.MultipleLocator(50))
        
        # for first array:
        #ax.yaxis.set_minor_locator(mtick.MultipleLocator(1))
        #ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
        # for second array:
        if i == 0:
            ax.yaxis.set_minor_locator(mtick.MultipleLocator(1))
            ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
        else:
            ax.set_title('Pulser Peaks')
            energies = ['1.000', '2.000', '3.000', '4.000', '5.000', '5.486', '6.000', '7.000']
            x_pos = [300, 520, 800, 1050, 1325, 1450, 1625, 1850]
            y_pos = [175, 140, 125, 130, 130, 150, 140, 140]
            for i, e in enumerate(energies):
                text = e + ' MeV'
                ax.text(x_pos[i], y_pos[i], text,)
            ax.yaxis.set_minor_locator(mtick.MultipleLocator(5))
            ax.yaxis.set_major_locator(mtick.MultipleLocator(25))
            ax.set_ylim(0, 200)
        ax.legend()
    ax.set_xlabel('Channel')
    plt.tight_layout()
    #plt.close()

def example_fityk():
    
    def split_gaussian(x, A, mu, hwhm1, hwhm2, C):
        # hwhm1 lies to the left
        sigma1 = hwhm1/np.sqrt(2*np.log(2))
        sigma2 = hwhm2/np.sqrt(2*np.log(2))
        x1 = x[x < mu]
        x2 = x[x >= mu]
        f1 = A * np.exp(-(x1 - mu)**2 / (2*sigma1**2)) + C
        f2 = A * np.exp(-(x2 - mu)**2 / (2*sigma2**2)) + C
        f = np.hstack([f1, f2])
        return f
    
    eg_hist = np.load('data/am241_0V_raw.npy')
    eg_channel = np.arange(1, len(eg_hist)+1)
    A = 10.65476
    mu = 1007.134
    hwhm1 = 61.48247
    hwhm2 = 56.30466
    sigma1 = hwhm1/np.sqrt(2*np.log(2))
    sigma2 = hwhm2/np.sqrt(2*np.log(2))
    C = 1.505377
    x = np.linspace(-mu, 3*sigma2, num=500) + mu
    
    fig, ax = plt.subplots()
    title = 'Supply Voltage: 0 V'
    ax.set_title(title)
    ax.plot(eg_channel, eg_hist, 'o', color='black', label='Data')
    ax.errorbar(eg_channel, eg_hist, yerr=np.sqrt(eg_hist), fmt=',', color='gray', label='Error')
    ax.plot(x, split_gaussian(x, A, mu, hwhm1, hwhm2, C), color='red', label='Fit')
    ax.set_xlim(50, 1250)
    ax.set_ylim(0, max(eg_hist) + 6)
    ax.set_ylabel('Counts')
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(50))
    ax.legend()
    plt.tight_layout()

def aluminum():
    thickness = np.array([0, 1.61, 3.00, 3.41])
    
    channel = np.array([1504.39, 1118.66, 704.347, 494.153])
    channel_s = np.array([0.727523, 1.17367, 2.06817, 3.87413])
    
    fwhm = np.array([30.8767, 49.1332, 66.676, 92.439]) * 2
    fwhm_s = np.array([0.645476, 1.15387, 2.14034, 2.76363]) * 2
    
    chisq = np.array([2005.1, 1818.8, 1961.01, 758.946])
    dof = np.array([1244, 1061, 804, 626])
    rechisq = chisq/dof
    pval = chi2.sf(chisq, dof) * 100
    
    ch2e = calibrate()
    
    energy, energy_s = ch2e(channel, channel_s) # MeV
    energy_lost = np.abs(energy - E_0)
    resolution, resolution_s = ch2e(fwhm, fwhm_s)
    
    fig, ax = plt.subplots(nrows=2, sharex=True)
    
    ax[0].errorbar(thickness, energy_lost, yerr=energy_s, color='black', fmt='.')
    ax[0].set_ylabel('Energy Lost [MeV]')
    ax[0].yaxis.set_major_locator(mtick.MultipleLocator(1))
    ax[0].yaxis.set_minor_locator(mtick.MultipleLocator(0.2))
    ax[0].set_ylim(0, 4)
    
    ax1 = ax[0].twinx()
    ax1.set_ylabel('Percent of Emission Energy [%]')
    ax1.set_ylim(0, 4/E_0*100)
    ax1.yaxis.set_major_locator(mtick.MultipleLocator(0.1*100))
    ax1.yaxis.set_minor_locator(mtick.MultipleLocator(0.02*100))
    
    ax[1].errorbar(thickness, resolution*1000, yerr=resolution_s*1000, fmt='.', color='black')
    ax[1].xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
    ax[1].yaxis.set_minor_locator(mtick.MultipleLocator(10))
    ax[1].set_ylabel('FWHM [KeV]')
    ax[1].set_xlabel(r'Thickness [mg/cm$^2$]')
    ax[1].set_xlim(-0.1, 3.6)
    #ax[1].set_ylim(0.09e3, 0.35e3)
    
    ax2 = ax[1].twinx()
    ax2.set_ylabel('Percent of Peak Energy [%]')
    ax2.set_ylim(0, 0.4*100)
    ax2.yaxis.set_major_locator(mtick.MultipleLocator(0.05*100))
    ax2.yaxis.set_minor_locator(mtick.MultipleLocator(0.01*100))
    
    plt.tight_layout()

def resolutions():
    voltage_supply = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]) # V

    total_resistance = 101 # M Ohm
    voltage_drop = np.array([0, 0.041, 0.084, 0.114, 0.135, 0.149, 0.157, 0.173, 0.184, 0.190, 0.198]) * total_resistance # V
    # 100 M Ohm from the preamp
    # 1 M Ohm from the resistor within the bias supply
    
    voltage_bias = voltage_supply - voltage_drop
    
    channels = np.array([1007.13, 1052.5, 1134.39, 1432.69, 1501.5, 1512.53, 1530.19, 1528.54, 1537.61, 1540.22, 1541.27])
    channels_s = np.array([4.92662, 4.84387, 5.18497, 2.74768, 1.82336, 2.14455, 1.76431, 1.78499, 1.57198, 1.91554, 1.66222])
    fwhm = np.array([56.3047, 53.1484, 54.5917, 21.9411, 14.514, 15.9458, 16.2379, 18.4155, 15.4405, 15.8362, 15.9164]) * 2
    fwhm_s = np.array([4.46536, 4.34674, 4.67711, 2.26512, 1.43433, 1.7409, 1.41964, 1.44555, 1.2377, 1.5518, 1.29353]) * 2
    
    pulser_ch = np.array([281.293, 559.814, 838.367, 1117.11, 1397.28, 1531.61, 1678.16, 1957.95])
    pulser_ch_s = np.array([0.0991482, 0.16209, 0.166326, 0.151908, 0.187174, 0.113161, 0.145929, 0.12916])
    pulser_fwhm = np.array([9.44854, 9.49268, 9.21001, 9.44922, 9.29329, 9.31303, 9.03433, 9.13585]) * 2
    pulser_fwhm_s = np.array([0.0996524, 0.128233, 0.125042, 0.124159, 0.125156, 0.136575, 0.141191, 0.136416]) * 2
    
    ch2e = calibrate()
    energy, energy_s = ch2e(channels, channels_s)
    resolution, resolution_s = ch2e(fwhm, fwhm_s)
    pulser_e, pulser_e_s = ch2e(pulser_ch, pulser_ch_s+10)
    pulser_e_s += -ch2e(10, 0)[0]
    pulser_res, pulser_res_s = ch2e(pulser_fwhm, pulser_fwhm_s+10)
    pulser_res_s += -ch2e(10, 0)[0]
    
    '''
    fig, ax = plt.subplots(nrows=2)
    ax[0].errorbar(energy, resolution, xerr=energy_s , yerr=resolution_s, fmt='.', color='black')
    ax[1].errorbar(pulser_e, pulser_res, xerr=pulser_e_s, yerr=pulser_res_s, fmt='.', color='black')
    '''
    fig, ax = plt.subplots()
    ax.errorbar(energy, resolution, xerr=energy_s , yerr=resolution_s, fmt='.', color='black', label='Detector')
    ax.errorbar(pulser_e, pulser_res, xerr=pulser_e_s, yerr=pulser_res_s, fmt='.', color='red', label='Pulser')
    ax.set_xlabel('Kinetic Energy [MeV]')
    ax.set_ylabel('FWHM [MeV]')
    ax.legend()
    plt.tight_layout()


D = 6.463E-03 # max distance of alpha in Si, CSDA
D_projected = 6.376E-03 # ", projected

voltage_supply = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]) # V

total_resistance = 101 # M Ohm
voltage_drop = np.array([0, 0.041, 0.084, 0.114, 0.135, 0.149, 0.157, 0.173, 0.184, 0.190, 0.198]) * total_resistance # V
# 100 M Ohm from the preamp
# 1 M Ohm from the resistor within the bias supply

voltage_bias = voltage_supply - voltage_drop

channels = np.array([1007.13, 1052.5, 1134.39, 1432.69, 1501.5, 1512.53, 1530.19, 1528.54, 1537.61, 1540.22, 1541.27])
channels_s = np.array([4.92662, 4.84387, 5.18497, 2.74768, 1.82336, 2.14455, 1.76431, 1.78499, 1.57198, 1.91554, 1.66222])

fwhm = np.array([56.3047, 53.1484, 54.5917, 21.9411, 14.514, 15.9458, 16.2379, 18.4155, 15.4405, 15.8362, 15.9164]) * 2
fwhm_s = np.array([4.46536, 4.34674, 4.67711, 2.26512, 1.43433, 1.7409, 1.41964, 1.44555, 1.2377, 1.5518, 1.29353]) * 2

energy_remaining = E_0 - energy
energy_remaining_s = energy_s

sorted_ind = np.argsort(np.abs(energy_remaining))

e2r = energy2range()
#d_r = e2r(np.abs(energy_remaining))

distance_remaining = np.array([6.393E-05, 7.591E-05, 8.322E-05, 1.019E-04, 1.108E-04, 1.878E-04, 
                               2.328E-04, 4.301E-04, 1.243E-03, 1.503E-03, 1.655E-03]) # low to high from NIST
distance_remaining_s = np.array([5.269E-05, 5.067E-05, 5.291E-05, 5.441E-05, 5.166E-05, 5.682E-05, 5.316E-05, 6.330E-05,
                                 8.840E-05, 8.493E-05, 8.569E-05]) # g/cm^2
    
d_copy = np.ones(len(distance_remaining))
d_s_copy = np.ones(len(distance_remaining))

for i, j in enumerate(sorted_ind):
    d_copy[j] = distance_remaining[i]
    d_s_copy[j] = distance_remaining_s[i]
distance_remaining = d_copy
distance_remaining_s = d_s_copy
del d_copy, d_s_copy

#distance_remaining = e2r(np.abs(energy_remaining))
#distance_remaining_s = e2r(energy_remaining_s)

distance = D - distance_remaining
distance_s = distance_remaining_s

# Convert to cm
distance = distance / density_si # cm
distance_s = distance_s / density_si

distance_sq = distance**2 # cm^2
distance_sq_s = distance_sq * 2 * (distance_s/distance)

def d2(V, c_0, c_1):
    return c_0 + c_1 * V

def const(V, c):
    return c

cutoff = 5

param2, error2 = curve_fit(d2, voltage_bias[:cutoff], distance_sq[:cutoff], sigma=distance_sq_s[:cutoff])
error2 = np.sqrt(np.diag(error2))

constant2, constant2_s = curve_fit(const, voltage_bias[cutoff:], distance_sq[cutoff:], sigma=distance_sq_s[cutoff:])
constant2_s = np.sqrt(np.diag(constant2_s))

chisq = np.sum(( (np.ones(len(distance_sq[cutoff:]))*constant2[0] - distance_sq[cutoff:]) / distance_sq_s[cutoff:] ) ** 2)
dof = len(voltage_bias[cutoff:]) - len(constant2)
rechisq = chisq/dof
pval = chi2.sf(chisq, dof) * 100
print('\nConstant GOF Test:')
print('Parameters:', constant2)
print('Errors:', constant2_s)
print('ChiSq:', chisq)
print('dof:', dof)
print('ChiSq/dof:', rechisq)
print('p-value:', pval, '%')

chisq = np.sum(( (d2(voltage_bias[:cutoff], param2[0], param2[1]) - distance_sq[:cutoff]) / distance_sq_s[:cutoff] ) ** 2)
dof = len(voltage_bias[:cutoff]) - len(param2)
rechisq = chisq/dof
pval = chi2.sf(chisq, dof) * 100

V_0 = param2[0]/param2[1]
V_0_s = V_0 * np.sqrt((error2[0]/param2[0])**2 + (error2[1]/param2[1])**2)
dmax = np.sqrt(constant2[0]) # cm
dmax_s = dmax * (1/2) * (error2[0]/constant2[0]) # cm

a = d_depleted**2 - param2[0]
da = error2[0]
b = param2[1]
db = error2[1]
V_depleted = a / b
V_depleted_s = V_depleted * np.sqrt((da/a)**2 + (db/b)**2)

N_d = 2 * epsilon / q / param2[1]
N_d_s = N_d * error2[1] / param2[1]

print('\nd^2 GOF Test:')
print('Parameters:', param2)
print('Errors:', error2)
print('ChiSq:', chisq)
print('dof:', dof)
print('ChiSq/dof:', rechisq)
print('p-value:', pval, '%')

print('\nResults:')
print('Intrinsic Voltage:', V_0, '+/-', V_0_s, 'V')
print('Depletion Voltage:', V_depleted, '+/-', V_depleted_s, 'V')
print('d_max:', dmax*1000*density_si, '+/-', dmax_s*1000*density_si, 'mg/cm^2')
print('N_d:', N_d, '+/-', N_d_s, '1/cm^3')

v_crit = (constant2[0] - param2[0]) / param2[1]
v = np.linspace(0, v_crit, num=200)
v2 = np.linspace(v_crit, max(voltage_bias)*2, num=200)
fig, ax = plt.subplots()

ax.errorbar(voltage_bias, distance_sq, yerr=distance_sq_s, fmt='.', color='black')
ax.plot(v, d2(v, param2[0], param2[1]), '--', color='gray')
ax.plot(v2, np.ones(len(v2))*constant2[0], '--', color='gray')

ax.set_xlabel('Voltage [V]')
ax.set_ylabel(r'd$^2$ [cm$^2$]')
ax.ticklabel_format(style = 'sci', scilimits = (0,0), axis = 'y', useMathText = True)
ax.set_ylim(0.4e-5,0.8e-5)
ax.set_xlim(-0.5, 32)
ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.1e-6))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5e-6))
ax.plot(v2, d2(v2, param2[0], param2[1]), ':', color='silver')
plt.tight_layout()
plt.close()

#aluminum stuffs
thickness = np.array([    3.0,     1.61,    3.41,   0 ]) #units of mg/cm^2
centerChannel = np.array([752.417, 1137.86, 516.092,1531.96 ])
channelError = np.array([ 4.807,   2.29,    5.939,1.64 ])

'''
def d(V, k, V_0):
    return k * np.sqrt(V + V_0)

param, error = curve_fit(d, voltage_bias[:cutoff], distance[:cutoff], sigma=distance_s[:cutoff])
error = np.sqrt(np.diag(error))

constant, constant_s = curve_fit(const, voltage_bias[cutoff:], distance[cutoff:], sigma=distance_s[cutoff:])
constant_s = np.sqrt(np.diag(constant_s))

V_0 = param[1]
V_0_s = error[1]
dmax = constant[0] # cm
dmax_s = dmax * (1/2) * (error[0]/constant[0]) # cm

a = d_depleted**2/param[0]**2
da = a * (error[0]/param[0])
b = param[1]
db = error[1]
V_depleted = a - b
V_depleted_s = np.sqrt(da**2 + db**2)

N_d = 2 * epsilon / q / param[0]**2
N_d_s = N_d * 2 * error[0] / param[0]

chisq = np.sum(( (d(voltage_bias[:cutoff], param[0], param[1]) - distance[:cutoff]) / distance_s[:cutoff] ) ** 2)
dof = len(voltage_bias[:cutoff]) - len(param)
rechisq = chisq/dof
pval = chi2.sf(chisq, dof) * 100
print('\nd GOF Test:')
print('Parameters:', param)
print('Errors:', error)
print('ChiSq:', chisq)
print('dof:', dof)
print('ChiSq/dof:', rechisq)
print('p-value:', pval, '%')

print('\nResults:')
print('Intrinsic Voltage:', V_0, '+/-', V_0_s, 'V')
print('Depletion Voltage:', V_depleted, '+/-', V_depleted_s, 'V')
print('d_max:', dmax*1000*density_si, '+/-', dmax_s*1000*density_si, 'mg/cm^2')
print('N_d:', N_d, '+/-', N_d_s, '1/cm^3')

v_crit = (constant[0]/param[0])**2 - param[1]
v = np.linspace(min(voltage_bias), v_crit, num=200)
v2 = np.linspace(v_crit, max(voltage_bias), num=200)
fig, ax = plt.subplots()
ax.errorbar(voltage_bias, distance, yerr=distance_s, fmt='.', color='black')
ax.plot(v, d(v, param[0], param[1]), '--', color='gray')
ax.plot(v2, np.ones(len(v2))*constant[0], '--', color='gray')
ax.set_xlabel('Voltage [V]')
ax.set_ylabel(r'Distance [g/cm$^2$]')
plt.tight_layout()
'''