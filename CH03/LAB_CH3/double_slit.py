import numpy as np
import matplotlib.pyplot as plt

dx = 0.005
L = 100
a = 0.03
d = 0.3
lambda_ = 0.000650
x = np.arange(-10, 10, dx)
y = np.arange(-1, 1, dx)
theta = np.arctan(x / L)

def sinc(x):
    output = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi != 0:
            output[i] = np.sin(xi) / xi
        else:
            output[i] = 1
    return output

# Compute sinc, cos, and Gaussian functions
sinc_func = sinc((np.pi * a * np.sin(theta)) / lambda_)**2
cos_func = np.cos((np.pi * d * np.sin(theta)) / lambda_)**2
gaussian = np.exp(-40 * (y**2))
intensity = sinc_func * cos_func

# Create the mesh grid for x and y
X, Y = np.meshgrid(x, y)

# Combine intensity and Gaussian
simula = np.outer(gaussian, intensity)

# Clip the values to the specified range
minima = 0
maxima = 0.05
simula = np.clip(simula, minima, maxima)

# Plot using contourf
plt.contourf(X, Y, simula, levels=500, cmap='hot', vmax=maxima, vmin=minima)
plt.colorbar()

plt.xlabel('$x$ (cm)')
plt.ylabel('$y$ (cm)')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Optional: Plot using imshow for comparison
plt.imshow(simula, cmap = 'hot')

plt.colorbar()
plt.xlabel('$x$ (cm)')
plt.ylabel('$y$ (cm)')
plt.show()
