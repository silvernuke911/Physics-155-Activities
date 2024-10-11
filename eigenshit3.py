import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Constants
hbar = 1.0  # natural units, ħ = 1
m = 1.0     # mass of the particle
L = 10.0    # width of the region where we solve Schrödinger equation
N = 1000    # number of grid points

# Set up the grid
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]

# Potential: Finite square well
V0 = 50.0  # Depth of the well
a = 1.0    # Half-width of the well

# Define the potential
V = np.zeros_like(x)
V[np.abs(x) > a] = 0
V[np.abs(x) <= a] = -V0

# Hamiltonian: Kinetic + Potential
T = -hbar**2 / (2 * m * dx**2) * (np.diag(np.ones(N-1), -1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), 1))
V_matrix = np.diag(V)
H = T + V_matrix

# Solve for eigenvalues and eigenfunctions
eigenvalues, eigenfunctions = eigh(H)

# Select the 3 lowest eigenvalues (bound states)
bound_states_indices = np.where(eigenvalues < 0)[0][:3]
bound_energies = eigenvalues[bound_states_indices]
bound_wavefunctions = eigenfunctions[:, bound_states_indices]

# Normalize the wavefunctions
for i in range(3):
    bound_wavefunctions[:, i] /= np.sqrt(np.trapz(bound_wavefunctions[:, i]**2, x))

# Plot the potential and eigenfunctions
plt.figure(figsize=(10, 6))
plt.plot(x, V, label='Potential $V(x)$', color='black', lw=2)
for i in range(3):
    plt.plot(x, bound_wavefunctions[:, i] + bound_energies[i], label=f'Eigenfunction {i+1}, E = {bound_energies[i]:.3f}')
plt.axhline(0, color='black', lw=1)
plt.title('Finite Square Well: Bound States')
plt.xlabel('$x$')
plt.ylabel('Energy and Eigenfunctions')
plt.legend()
plt.grid(True)
plt.show()

# Print eigenenergies
for i, energy in enumerate(bound_energies):
    print(f'Bound state {i+1} energy: E_{i+1} = {energy:.3f}')
