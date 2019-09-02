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

Then uses 6th Richardson Extrapolation to quickly converge to a
higher accuracy eigenvalue.

"""
# Importing libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg

alpha = 1/137
mc2 = 511e3 # eV

# Potential for QHM
def V(x):
    return np.abs(x)

n_states = 10 # Number of eigenstates to find
error_order = 6 # Error order for Richardson Extrapolation

Rmax = 11
Rmin = -11

#-----------------------------------------------------------------------------#
eigenvalues = np.ones((error_order//2, n_states))
for i in range(error_order // 2):
    N = 10000 * (2**i) # Number of mesh points to use
    h = (Rmax - Rmin) / N # Grid spacing
    
    # Define a space grid
    x = np.arange(Rmin+h, Rmax, h)
    
    d = np.ones((N-1)) * (1/h**2) + V(x) # Diagonal elements of the Hamiltonian
    e = np.ones((N-2)) * -1/(2 * h**2) # Offdiagonal elements of the Hamiltonian
    
    # Tridiagonal solver to find eigenvalues and eigenvectors
    eigval, eigvec = linalg.eigh_tridiagonal(d, e, select = 'i',
                                             select_range = (0,n_states-1))
    
    # Append obtained results for later extrapolation
    eigenvalues[i] = eigval[:n_states]

# Richardson Extrapolation
e1 = eigenvalues[1] + (eigenvalues[1] - eigenvalues[0]) / 3 # h 4th Order RE
e2 = eigenvalues[2] + (eigenvalues[2] - eigenvalues[1]) / 3 # h/2 4th Order RE

energies = e2 + (e2 - e1) / 3 # 6th Order RE
print(energies)
#energies = energies * mc2 * (alpha**2) # Convert to eV
#-----------------------------------------------------------------------------#
# Plotting commands
golden_ratio = (np.sqrt(5) + 1) / 2
width = 12
height = width / golden_ratio

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = (width, height)
'''
fig, ax = plt.subplots(nrows=n_states, sharex=True)
fig.suptitle('Eigenvectors: l = ' + str(l), weight='bold')
plt.subplots_adjust(left=0.1, right=0.95)
ax[2].set_xlabel('x')
for i in range(n_states):
    state = i + l + 1
    title = 'n = {}'.format(state)
    label = 'E = ' + str(np.round(energies[i], decimals=1)) + ' eV'
    ax[i].set_title(title)
    ax[i].set_ylabel(r'$\Psi (x)$')
    ax[i].set_xlim(Rmin, state * 10 + 10 * l)
    #ax[i].set_ylim(-0.1, 0.1)
    ax[i].grid()
    ax[i].plot(x, eigvec[:,i], color='black', label=label)
    ax[i].legend()
'''