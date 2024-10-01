import numpy as np
import matplotlib.pyplot as plt

# Define the potential function V(x)
def potential(x):
    return - x**3 + 5*x 
    # return np.sin(3*x)

# Define the Hamiltonian (total energy) H(x, p)
def hamiltonian(x, v, m):
    return 0.5 * m * v**2 + potential(x)

# Set parameters
m = 1.0  # mass

# Create a grid of x and p values
x = np.linspace(-5, 5, 500)
xdot = np.linspace(-5, 5, 500)
X, Xdot = np.meshgrid(x, xdot)

# Compute the Hamiltonian H(x, p) on the grid
H = hamiltonian(X, Xdot, m)

# Plot potential function
plt.plot(x,potential(x))
plt.grid()
plt.show()

# Plot phase space contours (constant energy curves)
plt.figure()
contour_levels = np.linspace(-20, 20, 40)  # Energy levels for contours
plt.contour(X, Xdot, H, levels=contour_levels, cmap = 'inferno')
plt.title('Phase Space Contours')
plt.xlabel('Position ($x$)')
plt.ylabel('Velocity ($v$)')
plt.grid(True)
plt.gca().set_aspect('equal')
plt.show()

plt.imshow(H)
