import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import matplotlib as mpl

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 15
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [9.0, 9.0]

# == 1 - Eigenvalues == #

def potential_infwell(x):
    f = x - x
    return f

def inf_well(L, n, x):
    f = np.sqrt(2/L)*np.sin(n*np.pi*x/L)
    return f

def schrodinger_1d(rmin, rmax, n, potential, numeigen):
    # rmin, rmax - Spatial limits
    # n - Number of steps
    # potential - potential function used
    # numeigen - number of eigenvalues desired
    
    h = (rmax - rmin) / n # Number of steps
    
    x = np.array([])
    for i in range(1,n):
        x = np.append(x, rmin + i*h)
    
    v = potential(x)
    
    d = v + 2/h**2
    
    e = np.array([])
    for i in range(n-2):
        e = np.append(e, -1/h**2)
    
    eigen = linalg.eigvalsh_tridiagonal(d, e, select = 'i', select_range = (0,numeigen-1)) / 2
    return eigen

def eigen_infwell(n, L): 
    # hbar = m = 1
    # n - integer values (eigen index)
    # L - length of well
    e = (n**2 * np.pi**2) / (2 * L**2) 
    return e

eigen_calc = schrodinger_1d(-np.pi, np.pi, 1000, potential_infwell, 5)
n = np.arange(1, 6, 1)
eigen_exact = eigen_infwell(n, 2*np.pi)

# == 2 - ODE == #

def func2(t, y, dy):
    f = -y
    return f

def func2_exact(t):
    f = np.cos(t)
    return f

def leapfrog(func, y0, dy0, a, b, n):
    t0 = a
    h = (b - a) / n
    dy_half = dy0 + func(t0, y0, dy0) * h /2
    
    y = np.array([y0])
    dy = np.array([dy_half])
    t = np.array([t0])
    for i in range(1,n+1):
        yi = y[-1] + h*dy[-1]
        dyi = dy[-1] + h*func(t[-1], yi, dy[-1])
        ti = t[-1] + h
        y = np.append(y, yi)
        dy = np.append(dy, dyi)
        t = np.append(t, ti)
    return y, dy, t

ode2_calc = leapfrog(func2, 1, 0, 0, 4, 100)
ode2_calc2 = leapfrog(func2, 1, 0, 0, 4, 200)
ode2_calc3 = leapfrog(func2, 1, 0, 0, 4, 400)
ode2_calc4 = leapfrog(func2, 1, 0, 0, 4, 800)
t2 = np.linspace(0, 4, 101)
ode2_exact = func2_exact(t2)

fig2, ax2 = plt.subplots()
ax2.plot(t2, ode2_calc[0], '+', markersize = 12, label = 'Leapfrog')
ax2.plot(t2, ode2_exact, color = 'orange', label = 'Exact')
ax2.set_ylabel('y')
ax2.set_xlabel('x')
ax2.set_title(r'$\frac{d^2}{dx^2}y = -y$')
ax2.grid()
ax2.legend()

# == 3 - ODE == #

def func3(t, y):
    f = y**2
    return f

def func3_exact(t):
    f = - 1 / (t - 1)
    return f

def rungekutta4(func, y0, a, b, n):
    # a - starting point
    # b - end point
    # n - step size
    # y0 - initial condition

    t0 = a
    h = (b - a) / n
    
    y = np.array([y0])
    t = np.array([t0])
    for i in range(1,n+1):
        k1 = h*func(t[-1], y[-1])
        k2 = h*func(t[-1] + h/2, y[-1] + k1/2)
        k3 = h*func(t[-1] + h/2, y[-1] + k2/2)
        k4 = h*func(t[-1] + h, y[-1] + k3)
        yi = y[-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        ti = t[-1] + h
        y = np.append(y, yi)
        t = np.append(t, ti)
    
    return y, t

ode3_calc = rungekutta4(func3, 1, 0, 1, 100)
t3 = np.linspace(0, 1, 101, endpoint = False)
ode3_exact = func3_exact(t3)

fig3, ax3 = plt.subplots()
ax3.plot(t3, ode3_calc[0], '+', markersize = 12, label = 'RK4')
ax3.plot(t3, ode3_exact, color = 'orange', label = 'Exact')
ax3.set_ylabel('y')
ax3.set_xlabel('x')
ax3.set_title(r'$\frac{d}{dx}y = y^2$')
ax3.grid()
ax3.legend()

# == 4 - ODE == #


# == Extra Credit == #