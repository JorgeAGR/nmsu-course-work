import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

def lsf(x, y, sx, sy, n=1):
	
	Y = np.zeros(shape = (n+1, 1))
	X = np.zeros(shape = (n+1, n+1))
	a = np.zeros(shape = (n+1, 1))
	i =0
	while True:
		i+=1
		dydx = 0
		A = a.copy()
		for i in range(len(a)):
			dydx += a[i][0] * i * x ** (i-1)#(i) * x ** (i-1)
		w = 1 / (sy ** 2 + (dydx * sx)**2)
		for i in range(n+1):
			for j in range(n+1):
				X[i][j] = np.sum(w * x**(j+i))
			Y[i][0] = np.sum(w * y * x ** i)
		a = np.dot(np.linalg.inv(X), Y)
		if np.abs(A - a).all() < 1e-6:
			print(i, np.abs(A- a))
			break
	
	return a.flatten()
	
	# JUST NEED THE UNCERTAINTIES!!
'''
	try:
		if not sx:
			sx = np.zeros(len(x))
	except:
		pass
	A = 0
	while True:
		a = A
		w = 1/(sy**2 + (A*sx)**2)
		S = np.sum(w)
		Sx = np.sum(w * x)
		Sy = np.sum(w * y)
		Sxx = np.sum(w * x**2)
		Sxy = np.sum(w * x * y)
		delta = S * Sxx - Sx ** 2
		A = (S * Sxy - Sx * Sy) / delta
		if np.abs(A-a) < 1e-10:
			break
	B = (Sxx * Sy - Sx * Sxy) / delta
	sA = np.sqrt(S / delta)
	sB = np.sqrt(Sxx / delta)
	chisq = np.sum((y - (A*x + B))**2 * w)
	rechisq = chisq / (len(x) - 2)
	
	print('A:',A,'+/-',sA)
	print('B:',B,'+/-',sB)
	print('ReChiSq:',rechisq)
	print(sx)
'''

def f(x):
	return 2 * x **2 + 8 * x - 10

def residual(vars, x, data, eps_data):
	
	a0 = vars[0]
	a1 = vars[1]
	a2 = vars[2]
	
	model = a0 + a1 * x + a2 * x ** 2 
	
	return (data - model) / eps_data

x = np.random.uniform(0, 4, 10)
sx = np.random.uniform(-0.01, 0.01, 10)
#sx = None
y_real = f(x)
sy = np.random.uniform(-1, 1, 10)
y = y_real + sy

#x = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
#				9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
#				16.0, 17.0,18.0,19.0,20.0,])
#sx = np.asarray([0.10, 0.20, 0.30, 0.40, 0.30, 0.20, 0.10,
#				 0.20, 0.30, 0.40, 0.30, 0.20, 0.10, 0.20, 
#				 0.30, 0.40, 0.30, 0.20, 0.10, 0.20])
#y = np.asarray([4.44, 4.10, 4.25, 4.06, 4.01, 3.61,
#				3.71, 3.37, 3.09, 3.14, 3.14, 3.09,
#				2.30, 2.30, 2.83, 2.71, 1.34, 1.95, 2.48, 0.69])
#y_real = y
#sy = np.asarray([0.12, 0.34, 0.14, 0.16, 0.16, 0.21,
#				 0.20, 0.25, 0.31, 0.30, 0.30, 0.31, 0.30,
#				 0.54, 0.37, 0.40, 0.50, 0.40, 0.47, 0.40])

mycoeff = lsf(x, y, sx, sy, n=2)
npcoeff = np.polyfit(x, y, 3)
spcoeff, _ = leastsq(residual, [0, 1, 1], args = (x, y, sy))

myfit = lambda x: mycoeff[0] + mycoeff[1] * x + mycoeff[2] * x ** 2
npfit = np.poly1d(npcoeff)
spfit = lambda x: mycoeff[0] + spcoeff[1] * x + spcoeff[2] * x ** 2

linspace = np.linspace(0,4)

plt.plot(linspace, f(linspace), 'o')
plt.errorbar(x, y, xerr = sx, yerr = sy, fmt = '.')
plt.plot(linspace, myfit(linspace), label = 'My')
plt.plot(linspace, npfit(linspace), '.-', label = 'Np')
plt.plot(linspace, spfit(linspace), '--', label = 'Sp')
plt.legend()
plt.show()