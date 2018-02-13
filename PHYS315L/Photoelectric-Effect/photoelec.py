import numpy as np
from uncertainties import ufloat
import uncertainties.umath as umath
import matplotlib.pyplot as plt

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
# == Plots == #
fig1, ax1 = plt.subplots()
ax1.plot(freqs, volt_avg_n, 'o')
'''
# == Intensisity Differences == #
measurementsGreen = {'100': , '80': , '60':, '40': , '20': }
measurementsBlue = {'100': , '80': , '60': , '40': , '20': }
'''