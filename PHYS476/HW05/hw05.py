import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 15
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [9.0, 9.0]

# == Problem 1 == #
'''
np.random.seed(1234)

n = np.arange(1, 21, 1)

r1_1 = np.random.random(10)
r1_2 = np.random.random(10)

r1 = np.append(r1_1, r1_2)

np.random.seed(1234)

r2 = np.random.random(20)

fig1, ax1 = plt.subplots()
fig1.suptitle('Drawing of random samples', fontweight = 'bold')
ax1.plot(n[:10], r1_1, 'x', markersize = 12, label = 'Draw #1, 1-10')
ax1.plot(n[10:], r1_2, 'x', markersize = 12, color = 'green', label = 'Draw #1, 11-20')
ax1.plot(n, r2, '+', markersize = 15, label = 'Draw #2, 1-20')
ax1.set_xlabel('n')
ax1.set_ylabel(r'$x_n$')
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.legend()
'''
# == Problem 2 == #

np.random.seed(1000)

def mc_pi(N):
    N = int(N)
    C = 0
    points = list(np.zeros(N))
    for i in range(N):
        x = np.random.random()
        y = np.random.random()
        points[i] = (x,y)
        if (x**2 + y**2) <= 1:
            C += 1
    return C*4/N, points

def coords(array):
    x = np.zeros(len(array))
    y = np.zeros(len(array))
    for i in range(len(array)):
        x[i] = array[i][0]
        y[i] = array[i][1]
    return x, y
        

n = np.array([1e1, 1e2, 1e3, 1e4,])

pi_calc = np.zeros(7)
points = list(np.zeros(7))
for i in range(len(n)):
    pi_calc[i], points[i] = mc_pi(n[i])

pi = dict(zip(n, pi_calc))

fig2, ax2 = plt.subplots(nrows = 2, ncols = 2)
fig2.suptitle(r'Calculating $\pi$', fontweight = 'bold')
circles = []
for n in range(4):
    circles.append(patches.Circle((0.5, 0.5), radius = 0.5, linewidth = 1, edgecolor = 'black', facecolor = 'none'))
plotpoints = (points[0], points[1], points[2], points[3])
i = 0
for axes in ax2:
    for ax in axes:
        p = coords(plotpoints[i])
        ax.plot(p[0], p[1], ',')
        ax.set_title('N = ' + str(len(p[0])))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(circles[i])
        i += 1
'''
fig2, ax2 = plt.subplots()
fig2.suptitle(r'Calculating $\pi$', fontweight = 'bold')
ax2.set_title(r'Approximiation of $\pi$ with increasing number of draws')
ax2.plot(pi.keys(), pi.values(), '--')
ax2.axhline(np.pi, color = 'black')
ax2.set_xscale('log')
ax2.set_xlabel('Numer of draws')
ax2.set_ylabel(r'\pi_{approx}')
ax2.annotate('Nominal Value', xy=(1e3, np.pi), xytext=(1e1, np.pi))

# == Problem 3 == #
np.random.seed(2000)

def f(x):
    return x**2

def g(x):
    return x

def mc(f, N):
    N = int(N)
    mcsum = 0
    #f = np.zeros(N)
    for i in range(N):
        x = 2*np.random.random()
        if f(x) >= 1:
            mcsum += f(x)
        #f[i] = x
    return mcsum/N

# == Problem 4 == #
np.random.seed(3000)



# == Problem 5 == #
np.random.seed(4000)

def f5(x):
    return 4/(1+x**2)

def mc_int(func, N):
    N = int(N)
    mcsum = 0
    for i in range(N):
        x = np.random.random()
        fx = func(x)
        mcsum += fx
    return mcsum/N

# == Problem 6 == #
np.random.seed(5000)

def mc_multint(func1, func2, func3, N):
    N = int(N)
    mcsum1 = mcsum2 = mcsum3 = 0
    for i in range(N):
        x1 = np.random.random()
        x2 = np.random.random()
        x3 = np.random.random()
        fx1 = func1(x1)
        fx2 = func2(x2)
        fx3 = func3(x3)
        mcsum1 += fx1
        mcsum2 += fx2
        mcsum3 += fx3
    return mcsum1*mcsum2*mcsum3/(N**3)
'''