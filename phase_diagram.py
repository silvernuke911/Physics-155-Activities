import numpy as np
import matplotlib.pyplot as plt

# Define the potential function V(x)
def potential(x):
    return - x**3 + 5*x 
    # return np.sin(3*x)

# Example 1: V(y) = 0.5 * y^2 (simple harmonic oscillator)
def potential_example_1(y1):
    return 0.5 * y1**2

# Example 2: V(y) = y^4 - y^2 (double-well potential)
def potential_example_2(y1):
    return y1**4 - y1**2

# Example 3: V(y) = y^3 (cubic potential)
def potential_example_3(y1):
    return -y1**3 + 5*y1

# Define the Hamiltonian (total energy) H(x, p)
def hamiltonian(x, v, m,V):
    return 0.5 * m * v**2 + V(x)

# Set parameters
m = 1.0  # mass

# Create a grid of x and p values
x = np.linspace(-5, 5, 500)
xdot = np.linspace(-5, 5, 500)
X, Xdot = np.meshgrid(x, xdot)

# Compute the Hamiltonian H(x, p) on the grid
H = hamiltonian(X, Xdot, m, potential_example_2)

# Plot potential function
plt.plot(x,potential_example_2(x))
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


