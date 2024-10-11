import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar_c = 197.3  # MeV * fm (conversion factor)
m = 469  # Reduced mass of proton-neutron system in MeV
E_target = -2.2  # Target bound state energy in MeV

# Potential well parameters (to be tuned)
V0 = 35  # Depth of the well in MeV (example value, tune this)
a = 2  # Width of the well in fm (example value, tune this)

# Discretization parameters
N = 1000  # Number of grid points
x_max = 10  # Maximum x value in fm
x = np.linspace(-x_max, x_max, N)
dx = x[1] - x[0]

# Potential function
def potential_well(x, a, V0):
    return np.where(np.abs(x) <= a, -V0, 0)

V = potential_well(x, a, V0)

# Kinetic energy operator (discretized second derivative)
T = -1 / (2 * m * dx**2) * (np.diag(np.ones(N - 1), -1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N - 1), 1))

# Total Hamiltonian (Kinetic + Potential)
H = T + np.diag(V)

# Solve for eigenvalues and eigenvectors using numpy's eigh
eigenvalues, eigenvectors = np.linalg.eigh(H)

# Find the ground state (first eigenvalue)
ground_state_energy = eigenvalues[0]
ground_state_wavefunction = eigenvectors[:, 0]

# Plot the potential and the wavefunction
plt.figure(figsize=(10, 6))

plt.plot(x, V, label='Potential Well $V(x)$')
plt.plot(x, ground_state_wavefunction**2 * 50, label='Probability Density $|\psi(x)|^2$', color='red')
plt.axhline(y=ground_state_energy, color='green', linestyle='--', label=f'Ground State Energy: {ground_state_energy:.2f} MeV')

plt.xlabel('x [fm]')
plt.ylabel('Energy [MeV] / Probability Density')
plt.title('Bound State in the Proton-Neutron Potential Well')
plt.legend()
plt.grid(True)
plt.show()

# Print the ground state energy
print(f'Ground state energy: {ground_state_energy:.2f} MeV')