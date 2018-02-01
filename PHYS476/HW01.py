import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

# 3

x = np.linspace(-3.0, 3.0, num = 50)
l = len(x)
nMax = 5 #Number of Hermite polynomials desired

X = sym.Symbol('x')
H = []
Hf = []

for n in range(nMax):
    h = sym.diff(sym.exp(-X**2), X, n) * ((-1)**n) * (sym.exp(X**2))
    H.append(h)
    Hf.append(sym.lambdify(X, H[n], 'numpy'))

hermiteAnalyticalMemory = np.zeros((nMax, l), dtype = int)
hermiteRecursionMemory = np.zeros((nMax, l), dtype = int)

for n in range(nMax):
    for i in range(l):
        hermiteAnalyticalMemory[n, i] = Hf[n](x[i])
        if n == 0:
                hermiteRecursionMemory[n, i] = 1
        elif n == 1:
                hermiteRecursionMemory[n, i] = 2 * x[i]
        else:
                hermiteRecursionMemory[n, i] = 2 * x[i] * hermiteRecursionMemory[n-1, i] - 2 * (n-1) * hermiteRecursionMemory[n-2, i]

fig = plt.figure(3)

plt.subplot2grid((2,2), (0,0))
for n in range(nMax):
    plt.plot(x, hermiteRecursionMemory[n, :])
plt.ylim(-25, 25)
plt.grid()
plt.title('Recursion H$_n$')

plt.subplot2grid((2,2), (0,1))
for n in range(nMax):
    plt.plot(x, hermiteAnalyticalMemory[n, :], '--')
plt.ylim(-50, 50)
plt.grid()
plt.title('Analytical H$_n$')

plt.subplot2grid((2,2),(1,0))
plt.plot(x, hermiteRecursionMemory[3, :], 'r')
plt.plot(x, hermiteAnalyticalMemory[3, :], 'b--')
plt.ylim(-50,50)
plt.grid()
plt.title('H$_3$')

plt.subplot2grid((2,2),(1,1))
plt.plot(x, hermiteRecursionMemory[4, :], 'r')
plt.plot(x, hermiteAnalyticalMemory[4, :], 'b--')
plt.ylim(-50,50)
plt.grid()
plt.title('H$_4$')

plt.show()

# 5a

x = 10

f = x**4

h = np.linspace(1, 0, num = 250, endpoint = False) 
for i in h:
    df2p = ((x+h)**4 - (x-h)**4)/(2*h)
    df4p = ((x-2*h)**4 - 8*(x-h)**4 + 8*(x+h)**4 - (x+2*h)**4)/(12*h)