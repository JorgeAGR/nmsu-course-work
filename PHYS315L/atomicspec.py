import numpy as np

def shiftTheta(measurement):
    shift = measurement['blue']
    for i in measurement:
        measurement[i] = np.abs(measurement[i] - shift)

def avgTheta():
    None

def deltaTheta(color):
    ((max(color) - min(color)) / 2 ) * (1 / (np.sqrt(len(color))))

def wavelength(d, theta, m):
    l = d * np.sin(theta) / m
    return l

d = 1/(600 / (10**(-3))) # meters
m1 = 1

measurements = [
{'blue': 139.4, 'violet': 124.4, 'green': 120.2, 'yellow': 119.1, 'orange': 118.9}, # Angle Patrick Right
{'blue': 139.4, 'violet': 154.5, 'green': 158.7, 'yellow': 159.7, 'orange': 159.9}, # Angle Patrick Left
{'blue': 139.3, 'violet': 124.1, 'green': 120.0, 'yellow': 119.0, 'orange': 118.9}, # Angle Jorge Right
{'blue': 139.3, 'violet': 154.5, 'green': 158.4, 'yellow': 159.5, 'orange': 159.6}, # Angle Jorge Left
{'blue': 139.4, 'violet': 124.3, 'green': 120.1, 'yellow': 119.0, 'orange': 118.9}, # Angle Ryan Right
{'blue': 139.4, 'violet': 154.5, 'green': 158.5, 'yellow': 159.6, 'orange': 159.7}, # Angle Ryan Left
]



for m in measurements:
    shiftTheta(m)

tv = []
tg = []
ty = []
to = []

for m in measurements:
    tv.append(m['violet'])
    tg.append(m['green'])
    ty.append(m['yellow'])
    to.append(m['orange'])

violet_t = np.mean(tv)
green_t = np.mean(tg)
yellow_t = np.mean(ty)
orange_t = np.mean(to)

lv = wavelength(d, violet_t, 1)