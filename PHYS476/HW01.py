import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3.0, 3.0, num = 50)
l = len(x)
nMax = 5 #Number of Hermite polynomials desired

numberMemory = np.zeros((nMax, l), dtype = int)

for n in range(nMax): 
    if n == 0:
        for i in range(l):
            numberMemory[n, i] = 1
    elif n == 1:
        for i in range(l):
            numberMemory[n, i] = 2 * x[i]
    else:
        for i in range(l):
            numberMemory[n, i] = 2 * x[i] * numberMemory[n-1, i] - 2 * (n-1) * numberMemory[n-2, i]

fig, ax = plt.subplots()
plt.ylim(-50, 50)

for n in range(nMax):
    plt.plot(x, numberMemory[n, :])

plt.show()