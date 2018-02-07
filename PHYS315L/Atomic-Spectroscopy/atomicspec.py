import numpy as np
from uncertainties import ufloat
import uncertainties.umath as umath

def shiftTheta(measurement):
    shift = measurement['blue']
    for i in measurement:
        measurement[i] = np.abs(measurement[i] - shift)

def avgTheta(measurement):
    tv = []
    tg = []
    ty = []
    to = []
    for m in measurement:
        tv.append(m['violet'])
        tg.append(m['green'])
        ty.append(m['yellow'])
        to.append(m['orange'])
    v = ufloat(np.mean(tv), np.std(tv))
    g = ufloat(np.mean(tg), np.std(tg))
    y = ufloat(np.mean(ty), np.std(ty))
    o = ufloat(np.mean(to), np.std(to))
    return v, g, y, o

def wavelength(d, theta, m):
    t = umath.radians(theta)
    l = d * umath.sin(t) / m
    return l

# == Mercury == #

d = 1/(600 / (10**(-3))) # meters
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
    shiftTheta(m)

theta_vHg_avg, theta_gHg_avg, theta_yHg_avg, theta_oHg_avg = avgTheta(measurementsHg)

vHg = wavelength(d, theta_v_avg, mHg)
gHg = wavelength(d, theta_g_avg, mHg)
yHg = wavelength(d, theta_y_avg, mHg)
oHg = wavelength(d, theta_o_avg, mHg)

# == Hydrogren == #

measurementsH = [
{'pink': 139.5, 'violet': 124.2, 'cyan': 122.2, 'red': 116.0, 'blue': 103.4}, # Angle Patrick Right
{'pink': , 'violet': , 'cyan': , 'red': , 'blue': }, # Angle Patrick Left
{'pink': , 'violet': , 'cyan': , 'red': , 'blue': }, # Angle Jorge Right
{'pink': , 'violet': , 'cyan': , 'red': , 'blue': }, # Angle Jorge Left
{'pink': 139.3, 'violet': 124.2, 'cyan': 122.3, 'red': 116.0, 'blue': 103.4}, # Angle Ryan Right
{'pink': 139.3, 'violet': 154.5, 'cyan': 156.3, 'red': 162.5, 'blue': 174.9}, # Angle Ryan Left