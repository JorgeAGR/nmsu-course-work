"""
Based on previous coursework done for PHYS 476
Reference: Hjorth-Jensen, Computational Physics (2012)

Solve Sch Eq for the QHM a number of times, which results
in different approximations for the eigenvalues for different
step sizes "h".

Then interpolate a function through these points using
Lagrangian Interpolation to obtain the value of the eigenstate
at h = 0. This results in a more accurate approximation while
saving time by solving smaller matrices.

It uses the approach of (after setting constants = 1) solving the
following form of the Sch. Eq.:

"""
# Importing libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.interpolate import lagrange

# Potential for QHM
def V(x):
    return 1/2 * x**2

n_states = 5 # Number of eigenstates to find
n_interp = 8 # Number of points for interpolation

eigenvalues = np.ones((n_interp,n_states))
h_vals = np.zeros(n_interp)
Rmax = 10
Rmin = -10
for i in range(n_interp):
    # Number of mesh points to use
    N = 10 * (2**i)
    h = (Rmax - Rmin) / N
    
    # Define a space grid
    x = np.arange(Rmin+h, Rmax, h)
    
    d = np.ones((N-1)) * (1/h**2) + V(x) # Diagonal elements of the Hamiltonian
    e = np.ones((N-2)) * -1/(2 * h**2) # Offdiagonal elements of the Hamiltonian
    
    # Tridiagonal solver to find eigenvalues and eigenvectors
    eigval, eigvec = linalg.eigh_tridiagonal(d, e, select = 'i',
                                             select_range = (0,n_states-1))
    
    # Append obtained results for interpolation later
    eigenvalues[i] = eigval[0:n_states]
    h_vals[i] = h

energies = np.zeros(n_states)
for i in range(n_states):
    # Interpolates and evaluates the interpolated function at 0
    # for better approximation of eigenvalues
    energies[i] = (lagrange(h_vals, eigenvalues[:, i])(0))

#----------------------------------------------------------------------------#
# Plotting commands
golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)

print(energies)
fig, ax = plt.subplots(nrows=n_states-2, sharex=True)
fig.suptitle('Eigenvectors', weight='bold')
plt.subplots_adjust(left=0.1, right=0.95)
ax[2].set_xlabel('x')
for i in range(n_states-2):
    title = 'n = {}'.format(i)
    ax[i].set_title(title)
    ax[i].set_ylabel(r'$\Psi (x)$')
    ax[i].set_xlim(-5, 5)
    ax[i].set_ylim(-0.1, 0.1)
    ax[i].grid()
    ax[i].plot(x, eigvec[:,i])