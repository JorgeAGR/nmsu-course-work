import numpy as np
from uncertainties import ufloat
import uncertainties.umath as umath
import matplotlib.pyplot as plt

def rydberg(n, i):
    # n = Landing state
    # i = Level jump
    R = 10973731.6
    l = 1 / ( R ( (1/n) - (1/(n+i)) ) )
    

def shiftTheta(measurement, center):
    shift = measurement[center]
    for i in measurement:
        measurement[i] = np.abs(measurement[i] - shift)

def avgTheta(measurement, color1, color2, color3, color4):
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    for m in measurement:
        t1.append(m[color1])
        t2.append(m[color2])
        t3.append(m[color3])
        t4.append(m[color4])
    c1 = ufloat(np.mean(t1), np.std(t1))
    c2 = ufloat(np.mean(t2), np.std(t2))
    c3 = ufloat(np.mean(t3), np.std(t3))
    c4 = ufloat(np.mean(t4), np.std(t4))
    return c1, c2, c3, c4

def wavelength(d, theta, m):
    t = umath.radians(theta)
    sint = umath.sin(t)
    l = d * sint / m
    return l, sint

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

theta_vHg_avg, theta_gHg_avg, theta_yHg_avg, theta_oHg_avg = avgTheta(measurementsHg, 'violet', 'green', 'yellow', 'orange')

vHg, sintvHg = wavelength(d, theta_vHg_avg, 1)
gHg, sintgHg = wavelength(d, theta_gHg_avg, 1)
yHg, sintyHg = wavelength(d, theta_yHg_avg, 1)
oHg, sintoHg = wavelength(d, theta_oHg_avg, 1)

sintheta = [sintvHg.n, sintgHg.n, sintyHg.n, sintoHg.n]
mlambda = [vHg.n, gHg.n, yHg.n, oHg.n]

figHg, axHg = plt.subplots()
axHg.errorbar(sintheta, mlambda, fmt = 'o')

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

theta_vH_avg, theta_cH_avg, theta_rH_avg, theta_bH_avg = avgTheta(measurementsH, 'violet', 'cyan', 'red', 'blue')

vH, sintvH = wavelength(d, theta_vH_avg, 1)
cH, sintcH = wavelength(d, theta_cH_avg, 1)
rH, sintrH = wavelength(d, theta_rH_avg, 1)
bH, sintbH = wavelength(d, theta_bH_avg, 2)

# == Uknown A == #
'''
measurementsA = [
{'center': , 'color1': , 'color2': , 'color3': , 'color4': }, # Angle Patrick Right
{'center': , 'color1': , 'color2': , 'color3': , 'color4': }, # Angle Patrick Left
{'center': , 'color1': , 'color2': , 'color3': , 'color4': }, # Angle Jorge Right
{'center': , 'color1': , 'color2': , 'color3': , 'color4': }, # Angle Jorge Left
{'center': , 'color1': , 'color2': , 'color3': , 'color4': }, # Angle Ryan Right
{'center': , 'color1': , 'color2': , 'color3': , 'color4': }, # Angle Ryan Left
'''

while (l < 700) and (l > 400):
    l = rydberg()