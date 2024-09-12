import numpy as np
import matplotlib.pyplot as plt

# Setting constants
num_points = 1000
mass = 1.0   # kg
length = 1.0 # m
g = 9.8      # m/s^2

# Setting plot space
theta = np.linspace(-2*np.pi, 2*np.pi, num_points)
theta_dot = np.linspace(-2*np.pi, 2*np.pi, num_points)
theta_grid, theta_dot_grid = np.meshgrid(theta, theta_dot)

# Function
E_surf = 0.5 * mass * length**2 * (theta_dot_grid **2) + mass * g * length * (1 -np.cos(theta_grid))

# Creating figure
fig = plt.figure(figsize=(6,6))
levels = 15
ax = fig.add_subplot()
contour = ax.contour(theta_grid , theta_dot_grid, E_surf, levels = levels , cmap = 'inferno')
values = ax.matshow(E_surf, alpha = 0.75, extent = [-2*np.pi, 2*np.pi,-2*np.pi, 2*np.pi], cmap='inferno')
fig.colorbar(values, ax=ax, shrink = 1, aspect = 15, label = 'Phase Contour', extend = "max", location = "right", cmap = 'inferno', ticks = np.arange(0,40,5))
ax.set_title('Contour Plot')
ax.set_aspect(1)
ax.set_xlabel(r'Angle position $\theta$')
ax.set_ylabel(r'Angle velocity $\dot{\theta}$')
plt.show()

