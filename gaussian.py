import numpy as np
import matplotlib.pyplot as plt

# Set parameters
dx = dy = 0.1
alpha = 0.1  # Thermal diffusivity (adjust as needed)
dt = 0.1
limits = [-5, 5]

x = np.arange(limits[0], limits[1], dx)
y = np.arange(limits[0], limits[1], dy)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Initial heat distribution
heat_func_0 = np.exp(-(R - 2.5)**2)

# Time range
t_final = 10
time_steps = int(t_final / dt)
t = np.linspace(0, t_final, time_steps)

# Neumann boundary condition handling for Laplacian
def laplacian_neumann(u, dx, dy):
    # """ Compute the Laplacian with Neumann boundary conditions (first derivative = 0 at edges). """
    # u_xx = np.zeros_like(u)
    # u_yy = np.zeros_like(u)
    
    # # Internal points
    # u_xx[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / dx**2
    # u_yy[1:-1, :] = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / dy**2
    
    # # Neumann boundary conditions (mirroring edges)
    # u_xx[:, 0] = (u[:, 1] - u[:, 0]) / dx**2  # Left boundary
    # u_xx[:, -1] = (u[:, -2] - u[:, -1]) / dx**2  # Right boundary
    # u_yy[0, :] = (u[1, :] - u[0, :]) / dy**2  # Top boundary
    # u_yy[-1, :] = (u[-2, :] - u[-1, :]) / dy**2  # Bottom boundary
    
    # # Ensure no NaNs
    # u_xx[np.isnan(u_xx)] = 0
    # u_yy[np.isnan(u_yy)] = 0

    # return u_xx + u_yy
    """ Compute the Laplacian of the 2D surface u. """
    d2x = np.gradient(np.gradient(u, axis=1), axis=1) / dx**2
    d2y = np.gradient(np.gradient(u, axis=0), axis=0) / dy**2
    return d2x + d2y


def heat_rhs(u, alpha, dx, dy):
    """ Compute the right-hand side of the heat equation. """
    return -alpha * laplacian_neumann(u, dx, dy)

def rk4_step(u, dt, alpha, dx, dy):
    """ Perform one step of the RK4 method for the heat equation. """
    k1 = heat_rhs(u, alpha, dx, dy)
    k2 = heat_rhs(u + 0.5 * dt * k1, alpha, dx, dy)
    k3 = heat_rhs(u + 0.5 * dt * k2, alpha, dx, dy)
    k4 = heat_rhs(u + dt * k3, alpha, dx, dy)
    
    return u + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Pre-allocate heat_series to store the time evolution
heat_series = np.zeros((time_steps, *heat_func_0.shape))

# Set the initial condition
heat_series[0] = heat_func_0
u = heat_func_0.copy()

# Evolve the system over time using RK4
for i in range(1, time_steps):
    u = rk4_step(u, dt, alpha, dx, dy)
    heat_series[i] = u

# Plot the initial and final states for comparison
plt.subplot(1, 2, 1)
plt.imshow(heat_func_0, cmap='hot', extent=[limits[0], limits[1], limits[0], limits[1]])
plt.colorbar()
plt.title('Initial Condition')

plt.subplot(1, 2, 2)
plt.imshow(heat_series[-1], cmap='hot', extent=[limits[0], limits[1], limits[0], limits[1]])
plt.colorbar()
plt.title('Final Condition')

plt.show()


# Form that has evolved over time. Balance your slides
