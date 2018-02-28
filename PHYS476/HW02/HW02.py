import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import lagrange
from scipy.misc import derivative as df

mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [9.0, 9.0]

def functest(x):
    f = 4*x - 7
    return f

def functest2(x):
    f = 3*x**3 + 5*x**2 - 8*x
    return f

def func3(x):
    f = x**3 - x
    return f

def dfunc3(x):
    f = 3 * x**2 - 1
    return f

def func4(x):
    f = x**2 - 4*x*np.sin(x) + 4*(np.sin(x))**2
    return f

def dfunc4(x):
    f = -2 * (x - 2*np.sin(x)) * (2*np.cos(x) - 1)
    return f

def hermite5(x):
    f = 32 * x ** 5 - 160 * x ** 3 + 120 * x
    return f

def dhermite5(x):
    f = 160 * x ** 4 - 480 * x ** 2 + 120
    return f

def hermite6(x):
    f = 64 * x ** 6 - 480 * x ** 4 + 720 * x ** 2 - 120
    return f

def dhermite6(x):
    f = 384 * x ** 5 - 1920 * x ** 3 + 1440 * x
    return f

def bisection(function, a, b):
    fa = function(a)
    fb = function(b)
    e = 10**(-5)
    d = e
    if (fa*fb > 0):
        print('Root not within interval')
        return
    while (np.abs(a - b) >= e):
        c = (a+b)/2
        fc = function(c)
        if (fc == 0) or (np.abs(fc) < d):
            break
        if (fa*fc < 0):
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c

def newtonraphson(func, dfunc, a, b, iterations):
    e = 10**(-5)
    d = e
    i = 0
    xn = (a + b) / 2
    while (i < iterations):
        if dfunc(xn) != 0:
            xn = xn - (func(xn) / dfunc(xn))
        else:
            print('Division by 0')
            return
        if (xn < a) or (xn > b):
            print('Out of bounds')
        elif (np.abs(func(xn)) < d):
            return xn
    print('Max iterations reached')

# == 1 - Interpolation == #
'''
x = np.linspace(-10,10, num = 250)

xtest = np.array([-3, 1, 8])
ytest = functest(xtest)
ftest = lagrange(xtest, ytest)

fig1a, ax1a = plt.subplots(2)
ax1a[0].set_title('Test: $f(x)=4x-7$')
ax1a[0].plot(x, functest(x), label = 'Actual $f(x)$')
ax1a[0].plot(xtest, ytest, 'o', label = '$x = -3, 1, 8$')
ax1a[0].grid()
ax1a[0].legend()
ax1a[1].plot(x, ftest(x), 'g', label = 'Interpolated $f(x)$')
ax1a[1].plot(xtest, ytest, 'o', color = 'xkcd:orange', label = '$x = -3, 1, 8$')
text1a = 'Coefficients:\n$a_0$: %f\n$a_1$: %f' % (ftest[0], ftest[1])
ax1a[1].text(2.5,-40, text1a, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})
ax1a[1].grid()
ax1a[1].legend()

xtest2 = np.array([-6, -1, 7, 9])
ytest2 = functest2(xtest2)
ftest2 = lagrange(xtest2, ytest2)

fig1b, ax1b = plt.subplots(2)
ax1b[0].set_title('Test: $f(x)=3x^3+5x^2-8x$')
ax1b[0].plot(x, functest2(x), label = 'Actual $f(x)$')
ax1b[0].plot(xtest2, ytest2, 'o', label = '$x = -6, -1, 7, 9$')
ax1b[0].grid()
ax1b[0].legend()
ax1b[1].plot(x, ftest2(x), 'g', label = 'Interpolated $f(x)$')
ax1b[1].plot(xtest2, ytest2, 'o', color = 'xkcd:orange', label = '$x = -6, -1, 7, 9$')
text1b = 'Coefficients:\n$a_0$: %e\n$a_1$: %f\n$a_2$: %f\n$a_3$: %f' % (ftest2[0], ftest2[1], ftest2[2], ftest2[3])
ax1b[1].text(7,-2000, text1b, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})
ax1b[1].grid()
ax1b[1].legend()

order = 5
x1 = np.array([-1.99991, 0.40541, 1.87182, 0.05990, -0.94838, -1.64181])
y1 = np.array([-0.74490, 1.67483, -0.43093, 0.19403, 1.23054, 0.68117])
f1 = lagrange(x1, y1)
xp = np.array([-1, 0, 1])
fx1 = f1(xp)

x = np.linspace(-2.1, 2)

def funcint(x):
    f = f1[5]*x**5 + f1[4]*x**4 + f1[3]*x**3 + f1[2]*x**2 + f1[1]*x + f1[0]
    return f

df1 = df(funcint, x, dx=1e-3)
dfx1 = df(funcint, xp, dx=1e-3)

fig1c, ax1c = plt.subplots()
ax1c.set_title('$P_5(x)$ Interpolation')
ax1c.plot(x1, y1, 'o', color = 'xkcd:orange', label = 'Data Points')
ax1c.plot(x, f1(x), 'g', label = '$P_5(x)$')
ax1c.plot(x, df1, label = 'd$P_5(x)$')
text1c1 = 'Coefficients:\n$a_0$: %f\n$a_1$: %f\n$a_2$: %f\n$a_3$: %f\n$a_4$: %f\n$a_5$: %f' % (f1[0], f1[1], f1[2], f1[3], f1[4], f1[5])
ax1c.text(-2,-6, text1c1, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})
text1c2 = '$f(-1)=%f$\n$f(0)=%f$\n$f(1)=%f$\n\n$df(-1)=%f$\n$df(0)=%f$\n$df(1)=%f$' % (fx1[0], fx1[1], fx1[2], dfx1[0], dfx1[1], dfx1[2])
ax1c.text(0,-6, text1c2, fontsize = 12, bbox = {'facecolor':'white','alpha':0.9,})
ax1c.set_ylim(-7.5,10)
ax1c.grid()
ax1c.legend()
'''
# == 2 - Linear Algebra == #

btest = np.array([19, 43])
Atest = np.matrix([[1,2],[3,4]])
xtest = np.linalg.solve(Atest,btest)
Ainvtest = np.linalg.inv(Atest)

btest2 = np.array([7,8,9])
Atest2 = np.matrix([[2,1,1],[1,2,1],[1,1,2]])
xtest2 = np.linalg.solve(Atest2,btest2)
Ainvtest2 = np.linalg.inv(Atest2)

b = np.array([-0.74982,-0.55492,-0.26831,-0.70489,0.52258,0.92891,0.22761,-0.34004,0.75930,-0.18726])
row1 = [-0.69745,-0.82230,0.89772,0.08127,-0.91479,0.09995,0.47018,-0.03091,-0.06901,-0.98549]
row2 = [-0.58764,0.14297,-0.76998,0.20914,-0.65973,0.39600,-0.51194,0.32937,-0.91729,-0.65270]
row3 = [0.50299,-0.21324,0.64983,0.12696,0.50619,0.38519,-0.36190,0.65688,0.06510,0.56082]
row4 = [0.17067,0.58526,0.90487,0.82506,0.56727,0.58876,0.14412,-0.98118,-0.69416,0.02011]
row5 = [-0.40881,0.20290,0.11984,0.69425,0.00835,0.82289,-0.37997,0.58696,0.96516,-0.87272]
row6 = [-0.55557,-0.10287,0.22758,-0.39297,0.83489,-0.88758,-0.52796,0.60384,-0.12730,0.96763]
row7 = [-0.89787,0.68322,-0.52039,0.10452,-0.79060,0.89818,0.06959,-0.64246,0.04281,0.67042]
row8 = [-0.11979,-0.36472,0.41455,0.95572,-0.59372,0.47017,-0.27596,0.90723,-0.93159,-0.90106]
row9 = [0.14508,0.96358,-0.87831,-0.86274,0.62326,-0.76172,0.95837,-0.38542,-0.54835,0.63309]
row10 = [0.72396,0.23809,0.96664,0.50911,-0.62429,0.83198,0.33306,-0.92702,0.03403,0.66115]

A = np.matrix([row1, row2, row3, row4, row5, row6, row7, row8, row9, row10])
x2 = np.linalg.solve(A, b)
Ainv = np.linalg.inv(A)

# == 3 - f(x) = x^3 - x == #

x = np.linspace(-3, 3, num = 250)
'''
a3_1 = np.linspace(-1.9, -1.1)
b3_1 = np.linspace(-0.1, -0.9)
f3r_bi1 = []
for i in range(len(a3_1)):
    f = bisection(func3, a3_1[i], b3_1[i])
    f3r_bi1.append(f)
f3r_nr1 = []
for i in range(len(a3_1)):
    f = newtonraphson(func3,dfunc3,a3_1[i],b3_1[i],10)
    f3r_nr1.append(f)

a3_2 = np.linspace(-0.9,-0.1)
b3_2 = np.linspace(0.9, 0.1)
f3r_bi2 = []
for i in range(len(a3_2)):
    f = bisection(func3, a3_2[i], b3_2[i])
    f3r_bi2.append(f)
f3r_nr2 = []
for i in range(len(a3_2)):
    f = newtonraphson(func3,dfunc3,a3_2[i],b3_2[i],10)
    f3r_nr2.append(f)

a3_3 = np.linspace(0.1,0.9)
b3_3 = np.linspace(1.9, 1.1)
f3r_bi3 = []
for i in range(len(a3_3)):
    f = bisection(func3, a3_3[i], b3_3[i])
    f3r_bi3.append(f)
f3r_nr3 = []
for i in range(len(a3_3)):
    f = newtonraphson(func3,dfunc3,a3_3[i],b3_3[i],10)
    f3r_nr3.append(f)

f3r_bi = np.array([bisection(func3, -1.5, -0.6), bisection(func3, -0.6, 0.6), bisection(func3, 0.6, 1.5)])
f3v_bi = func3(f3r_bi)

f3r_nr = np.array([newtonraphson(func3, dfunc3, -1.5, -0.6, 10),
                   newtonraphson(func3, dfunc3, -0.6, 0.6, 10),
                   newtonraphson(func3, dfunc3, 0.6, 1.5, 10)])
f3v_nr = func3(f3r_nr)

fig3a, ax3a = plt.subplots()
ax3a.plot(x, func3(x), label = 'Plot')
ax3a.plot(f3r_bi, f3v_bi, 'og', label = 'Bisection')
ax3a.plot(f3r_nr, f3v_nr, 'xr', markersize = 12, label = 'Newton-Raphson')
ax3a.set_title('$f(x) = x^3 - x$')
ax3a.set_xlabel('$x$')
ax3a.set_ylabel('$f(x)$')
ax3a.set_ylim(-1,1)
ax3a.set_xlim(-1.5,1.5)
ax3a.grid()
ax3a.legend()

fig3b, ax3b = plt.subplots()
ax3b.plot(b3_1-a3_1, f3r_bi1, label = '$x = -1$')
ax3b.plot(b3_2-a3_2, f3r_bi2, label = '$x = 0$')
ax3b.plot(b3_3-a3_3, f3r_bi3, label = '$x = 1$')
ax3b.set_title('Bisection \nInterval Size vs Calculated Root')
ax3b.set_ylabel('Root $x_0$')
ax3b.set_xlabel('Interval $(b - a)$')
ax3b.set_ylim(-1.5,1.5)
ax3b.invert_xaxis()
ax3b.grid()
ax3b.legend()

fig3c, ax3c = plt.subplots()
ax3c.plot(b3_1-a3_1, f3r_nr1, label = '$x = -1$')
ax3c.plot(b3_2-a3_2, f3r_nr2, label = '$x = 0$')
ax3c.plot(b3_3-a3_3, f3r_nr3, label = '$x = 1$')
ax3c.set_title('Newton-Raphson \nInterval Size vs Calculated Root')
ax3c.set_ylabel('Root $x_0$')
ax3c.set_xlabel('Interval $(b - a)$')
ax3c.set_ylim(-1.5,1.5)
ax3c.invert_xaxis()
ax3c.grid()
ax3c.legend()
'''
# == 4 - f(x) = x^2 - 4xsinx + 4(sinx)^2 == #
'''
a4_1 = np.linspace(-2.4,-1.9)
b4_1 = np.linspace(-1.3,-1.8)
f4r_bi1 = []
for i in range(len(a4_1)):
    f = bisection(func4, a4_1[i], b4_1[i])
    f4r_bi1.append(f)
f4r_nr1 = []
for i in range(len(a4_1)):
    f = newtonraphson(func4,dfunc4,a4_1[i],b4_1[i],10)
    f4r_nr1.append(f)

a4_2 = np.linspace(-0.7,-0.2)
b4_2 = np.linspace(0.6, 0.1)
f4r_bi2 = []
for i in range(len(a4_2)):
    f = bisection(func4, a4_2[i], b4_2[i])
    f4r_bi2.append(f)
f4r_nr2 = []
for i in range(len(a4_2)):
    f = newtonraphson(func4,dfunc4,a4_2[i],b4_2[i],10)
    f4r_nr2.append(f)

a4_3 = np.linspace(1.3,1.8)
b4_3 = np.linspace(2.4,1.9)
f4r_bi3 = []
for i in range(len(a4_1)):
    f = bisection(func4, a4_3[i], b4_3[i])
    f4r_bi3.append(f)
f4r_nr3 = []
for i in range(len(a4_3)):
    f = newtonraphson(func4,dfunc4,a4_3[i],b4_3[i],10)
    f4r_nr3.append(f)

f4r_bi = np.array([bisection(func4, -2.0, -1.5), bisection(func4, -0.6, 0.6), bisection(func4, 1.5, 2.0)])

f4r_nr = np.array([newtonraphson(func4, dfunc4, -2.0, -1.5, 10),
                   newtonraphson(func4, dfunc4, -0.5, 0.6, 10),
                   newtonraphson(func4, dfunc4, 1.5, 2.0, 10)])
f4v_nr = func4(f4r_nr)

fig4a, ax4a = plt.subplots()
ax4a.plot(x, func4(x), label = 'Plot')
ax4a.plot(f4r_nr, f4v_nr, 'xr', markersize = 12, label = 'Newton-Raphson')
ax4a.set_title('$f(x) = x^2 - 4xsinx + 4sin^2x$')
ax4a.set_xlabel('$x$')
ax4a.set_ylabel('$f(x)$')
ax4a.set_ylim(None,1)
ax4a.set_xlim()
ax4a.grid()
ax4a.legend()

fig4c, ax4c = plt.subplots()
ax4c.plot(b4_1-a4_1, f4r_nr1, label = '$x = -1.894$')
ax4c.plot(b4_2-a4_2, f4r_nr2, label = '$x = 0$')
ax4c.plot(b4_3-a4_3, f4r_nr3, label = '$x = 1.894$')
ax4c.set_title('Newton-Raphson \nInterval Size vs Calculated Root')
ax4c.set_ylabel('Root $x_0$')
ax4c.set_xlabel('Interval $(b - a)$')
ax4c.set_ylim(-2,2)
ax4c.invert_xaxis()
ax4c.grid()
ax4c.legend()
'''

# == 5 - Hermite Polynomials 5 & 6 == #

h5r_bi = np.array([bisection(hermite5,-2.5,-2.0), bisection(hermite5, -1.2, -0.7),
                   bisection(hermite5, -0.3, 0.2), bisection(hermite5, 0.7, 1.2),
                   bisection(hermite5, 2.0, 2.5)])
h5v_bi = hermite5(h5r_bi)

h5r_nr = np.array([newtonraphson(hermite5, dhermite5, -2.5, -2.0, 10),
                   newtonraphson(hermite5, dhermite5, -1.2, -0.7, 10),
                   newtonraphson(hermite5, dhermite5, -0.3, 0.2, 10),
                   newtonraphson(hermite5, dhermite5, 0.7, 1.2, 10),
                   newtonraphson(hermite5, dhermite5, 2.0, 2.5, 10)])
h5v_nr = hermite5(h5r_nr)

h6r_bi = np.array([bisection(hermite6,-3.0,-2.3), bisection(hermite6,-1.6,-1.0),
                   bisection(hermite6, -0.6, -0.2), bisection(hermite6, 0.2, 0.6),
                   bisection(hermite6, 1.0, 1.6), bisection(hermite6, 2.3, 3.0)])
h6v_bi = hermite6(h6r_bi)

h6r_nr = np.array([newtonraphson(hermite6, dhermite6, -3.0, -2.3, 10),
                   newtonraphson(hermite6, dhermite6, -1.6, -1.0, 10),
                   newtonraphson(hermite6, dhermite6, -0.6, -0.2, 10),
                   newtonraphson(hermite6, dhermite6, 0.2, 0.6, 10),
                   newtonraphson(hermite6, dhermite6, 1.0, 1.6, 10),
                   newtonraphson(hermite6, dhermite6, 2.3, 3.0, 10)])
h6v_nr = hermite6(h6r_nr)

fig5a, ax5a = plt.subplots()
ax5a.plot(x, hermite5(x), label = 'Plot')
ax5a.plot(h5r_bi, h5v_bi, 'og', label = 'Bisection')
ax5a.plot(h5r_nr, h5v_nr, 'xr', markersize = 12, label = 'Newton-Raphson')
ax5a.set_title('$H_5(x) = 32x^5 - 160x^3 + 120x$')
ax5a.set_xlabel('$x$')
ax5a.set_ylabel('$f(x)$')
ax5a.set_ylim(-175,175)
#ax5a.set_xlim()
ax5a.grid()
ax5a.legend()

fig6a, ax6a = plt.subplots()
ax6a.plot(x, hermite6(x), label = 'Plot')
ax6a.plot(h6r_bi, h6v_bi, 'og', label = 'Bisection')
ax6a.plot(h6r_nr, h6v_nr, 'xr', markersize = 12, label = 'Newton-Raphson')
ax6a.set_title('$H_6(x) = 64x^4 - 480x^4 + 720x^2 - 120$')
ax6a.set_xlabel('$x$')
ax6a.set_ylabel('$f(x)$')
ax6a.set_ylim(-900,900)
#ax5a.set_xlim()
ax6a.grid()
ax6a.legend()
