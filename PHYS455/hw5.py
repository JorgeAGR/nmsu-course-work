# Importing libraries
import numpy as np
from scipy import linalg

# This function creates the matrix for H = H_0 + dV for any input size N
def qho_H(N):
    energies_w0 = np.ones(N) # Array of ones of size N
    potential_dw = np.zeros((N,N)) # Matrix of zeros of size NxN
    for n in range(N):
        energies_w0[n] = (2 * n + 1) / 2 * np.sqrt(2) # Defines the elements for H_0
        potential_dw[n][n] = (2*n + 1) / 4 * 0 # Defines the diagonal for dV
        if n < (N - 2):
            # Off-diagonal elements above the diagonal
            potential_dw[n][n+2] = np.sqrt((n + 1) * (n + 2)) / 4 * 0
        if n >= 2:
            # Off-diagonal elements below the diagonal
            potential_dw[n][n-2] = np.sqrt(n * (n - 1)) / 4 * 0
    # Diagonalizes H_0 elements and adds both matrices
    return np.diag(energies_w0) + potential_dw

# Part 3
# Simply printing sample matrices to prove the previous function works correctly
print('Generate Matrix H Samples:')
for i in range(3):
    print(i+2, 'x', i+2, 'Matrix')
    print(qho_H(i + 2), '\n')
print('\n', '\n')

# Part 4
# Solves for the eigenvalues of various dimensions and calculates their average % error
print('Solve for Eigenvalues of H:')
N = [4, 10, 100]
for n in N:
    h = qho_H(n)
    e = linalg.eigvalsh(h)
    avg_err = 0
    if n == 4:
        print('Lowest 4 Eigenvalues,  Matrix Dim:', n, 'x', n)
        print(e[:4])
        for i in range(4):
            avg_err += (e[i] - (2*i + 1)/2) / (2*i + 1)
        avg_err = avg_err / 4
        print(avg_err*100, '% Avg Error', '\n')
    else:
        print('Lowest 10 Eigenvalues, Matrix Dim:', n, 'x', n)
        print(e[:10])
        for i in range(10):
            avg_err += (e[i] - (2*i + 1)/2) / (2*i + 1)
        avg_err = avg_err / 10
        print(avg_err*100, '% Avg Error', '\n')