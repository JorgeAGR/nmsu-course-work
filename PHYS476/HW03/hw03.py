import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def func2(x):
    f = 1 / (1 + x**2)
    return f

def func3(y):
    f = 1 / (1 - 2 * y + 2 * y**2 )
    return f

def func4(x):
    f = np.cos(x) / np.sqrt(1 - (1/2)**2 * (np.sin(x))**2)
    return f

def func5(x):
    f = np.cos(x) / (1 + x**2)
    return f

def trapezoidal(a, b, n, f):
    step = (b - a) / n
    fa = f(a)
    fb = f(b)
    trapsum = 0
    for i in range(1,n):
        x = i * step + a
        trapsum += f(x)
    trapsum = ( (fa / 2) + (fb / 2) + trapsum ) * step
    return trapsum

def simpsons(a, b, n, f):
    step = (b - a) / n
    fa = f(a)
    fb = f(b)
    simpsum = 0
    for i in range(1,n):
        x = i * step + a
        if i % 2 == 0:
            simpsum += 2 * f(x)
        else:
            simpsum += 4 * f(x)
    simpsum = (simpsum + fa + fb) * step / 3
    return simpsum

def gauss(a, b, n, f):
    x, w = np.polynomial.legendre.leggauss(n)
    i = ((b - a) / 2 ) * np.sum(f(((b-a)*x/2 + (b+a)/2)) * w )
    return i

def gaussinf(n, f):
    x, w = np.polynomial.legendre.leggauss(n)
    x2 = np.tan((np.pi/4) * (1+x))
    w2 = (np.pi / 4) * (w / (np.cos( (np.pi/4) * (1+x) ) )**2)
    i = np.sum(w2*f(x2))
    return i

def accuracy(k, integral, a, b, f, exact, error):
    n = 1
    i = k * integral(a, b, n, f)
    e = np.abs(1 - i / exact)
    while e > error:
        n += 1
        i = k * integral(a, b, n, f)
        e = np.abs(1 - i / exact)
    return i, n

def accuracyinf(k, integral, f, exact, error):
    n = 1
    i = k * integral(n, f)
    e = np.abs(1 - i / exact)
    while e > error:
        n += 1
        i = k * integral(n, f)
        e = np.abs(1 - i / exact)
    return i, n

# == Problem 1 == #
testmatrix = np.matrix([
        [6, -1],
        [2, 3]
        ])

testeval, testevec = np.linalg.eig(testmatrix)

matrix = np.matrix([
        [0.00009, 0.34013, 2.40541, 3.56645, 3.87182],
        [0.34013, 0.75876, 2.05990, 1.59203, 1.05162],
        [2.40541, 2.05990, 2.97405, 0.35819, 2.24156],
        [3.56645, 1.59203, 0.35819, 2.32892, 3.23827],
        [3.87182, 1.05162, 2.24156, 3.23827, 2.36768]
        ])

eigenvalues, eigenvectors = np.linalg.eig(matrix)

# == Problem 2 == #
#integ2_trap = 4 * trapezoidal(0, 1, 100, func2)
#integ2_simp = 4 * simpsons(0, 1, 100, func2)
#integ2_gauss = 4 * gauss(0, 1, 100, func2)
integ2_trap = {1: accuracy(4, trapezoidal, 0, 1, func2, np.pi, 0.01),
               0.1: accuracy(4, trapezoidal, 0, 1, func2, np.pi, 0.001), 
               0.01: accuracy(4, trapezoidal, 0, 1, func2, np.pi, 0.0001)}
integ2_simp = {1: accuracy(4, simpsons, 0, 1, func2, np.pi, 0.01),
               0.1: accuracy(4, simpsons, 0, 1, func2, np.pi, 0.001), 
               0.01: accuracy(4, simpsons, 0, 1, func2, np.pi, 0.0001)}
integ2_gauss = {1: accuracy(4, gauss, 0, 1, func2, np.pi, 0.01),
                0.1: accuracy(4, gauss, 0, 1, func2, np.pi, 0.001), 
                0.01: accuracy(4, gauss, 0, 1, func2, np.pi, 0.0001)}

# == Problem 3 == #
integ3_trap = {1: accuracy(1, trapezoidal, 0, 1, func3, np.pi / 2, 0.01),
               0.1: accuracy(1, trapezoidal, 0, 1, func3, np.pi / 2, 0.001), 
               0.01: accuracy(1, trapezoidal, 0, 1, func3, np.pi / 2, 0.0001)}
integ3_simp = {1: accuracy(1, simpsons, 0, 1, func3, np.pi / 2, 0.01),  
               0.1: accuracy(1, simpsons, 0, 1, func3, np.pi / 2, 0.001), 
               0.01: accuracy(1, simpsons, 0, 1, func3, np.pi / 2, 0.0001)}
integ3_gauss = {1: accuracy(1, gauss, 0, 1, func3, np.pi / 2, 0.01),
                0.1: accuracy(1, gauss, 0, 1, func3, np.pi / 2, 0.001), 
                0.01: accuracy(1, gauss, 0, 1, func3, np.pi / 2, 0.0001)}

# == Problem 4 == #
integ4_trap = {1: accuracy(1, trapezoidal, 0, np.pi/2, func4, np.pi / 3, 0.01),
               0.1: accuracy(1, trapezoidal, 0, np.pi/2, func4, np.pi / 3, 0.001), 
               0.01: accuracy(1, trapezoidal, 0, np.pi/2, func4, np.pi / 3, 0.0001)}
integ4_simp = {1: accuracy(1, simpsons, 0, np.pi/2, func4, np.pi / 3, 0.01),  
               0.1: accuracy(1, simpsons, 0, np.pi/2, func4, np.pi / 3, 0.001), 
               0.01: accuracy(1, simpsons, 0, np.pi/2, func4, np.pi / 3, 0.0001)}
integ4_gauss = {1: accuracy(1, gauss, 0, np.pi/2, func4, np.pi / 3, 0.01),
                0.1: accuracy(1, gauss, 0, np.pi/2, func4, np.pi / 3, 0.001), 
                0.01: accuracy(1, gauss, 0, np.pi/2, func4, np.pi / 3, 0.0001)}

x = np.arange(1, 21, 1)
sx = []
for i in x:
    s = simpsons(0, np.pi/2, i, func3)
    sx.append(s)
tx = []
for i in x:
    t = trapezoidal(0, np.pi/2, i, func3)
    tx.append(t)
gx = []
for i in x:
    g = gauss(0, np.pi/2, i, func3)
    gx.append(g)

plt.plot(x, sx)
plt.plot(x,tx)
plt.plot(x,gx)
# == Problem 5 == #
integ5_gauss = accuracyinf(1, gaussinf, func5, np.pi / (2*np.exp(1)), 0.0001)

