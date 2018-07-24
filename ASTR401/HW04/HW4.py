import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [9.0, 9.0]

def prob2(tau, N):
    I = [1]
    dt = tau/N
    for n in range(1,N):#range(1,N+1):
        iout = I[-1] - I[-1]*dt
        I.append(iout)
    return np.array(I), I[-1]

def func(tau):
    iratio = np.exp(-tau)
    return iratio

tau = np.array([0.1, 1, 10])
N = 1000

I1, end1= prob2(tau[0], N)
I2, end2 = prob2(tau[1], N)
I3, end3 = prob2(tau[2], N)

fig, ax = plt.subplots()
ax.plot(range(N),I1)#, label = r'$\tau = 0.1$')
ax.plot(range(N),I2, '-.')#, label = r'$\tau = 1$')
ax.plot(range(N),I3, '--')#, label = r'$\tau = 10$')
ax.plot((1000,1000,1000), func(tau), 'o', markersize = 10, label = r'$I/I_0=e^{-\tau}$')
ax.set_xlim(0, N)
ax.set_ylabel('Fraction Transmitted')
ax.set_xlabel('Layer #')
ax.grid()
ax.legend()
ax.annotate(r'$\tau = 0.1$', xy=(600, 0.95), xytext=(600, 0.95))
ax.annotate(r'$\tau = 1$', xy=(500, 0.62), xytext=(500, 0.62))
ax.annotate(r'$\tau = 10$', xy=(400, 0.03), xytext=(400, 0.03))