import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x_grid = np.linspace(-1, 3, 100)
y_grid = np.linspace(-2, 1.5, 100)

X, Y = np.meshgrid(x_grid, y_grid)
phi = X**2 - 2*X + Y**4 - 2*Y**2 + Y

ms = ax.matshow(phi, vmin=-3, vmax=0.6, origin="lower", alpha=0.3, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()])
cp = ax.contour(X, Y, phi, levels=[-3, -2.4, -1.8, -1, -0.5, 0.6])

bgcb = fig.colorbar(ms, ax=ax)
fig.colorbar(cp, cax=bgcb.ax)
plt.show()

dx = np.diff(x_grid)[0]
dy = np.diff(y_grid)[0]

# Compute partial derivative dphi/dx using forward difference
dphi_dx = (np.roll(phi, -1, axis=1) - phi) / dx
dphi_dy = (np.roll(phi, -1, axis=0) - phi) / dy

dphi_dx = dphi_dx[1:-1,1:-1]
dphi_dy = dphi_dy[1:-1,1:-1]

# Plotting the partial derivatives
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot for dphi / dx
# c1 = ax[0].contour(X, Y, dphi_dx, cmap="coolwarm")
c1 = ax[0].matshow(dphi_dx, cmap="coolwarm")
fig.colorbar(c1, ax=ax[0])
ax[0].set_title(r"Partial Derivative $\partial \phi /\partial x$")

# Plot for dphi / dy
c2 = ax[1].matshow(dphi_dy, cmap="coolwarm")
fig.colorbar(c2, ax=ax[1])
ax[1].set_title(r"Partial Derivative $\partial \phi /\partial y$")

plt.show()

# Compute second derivative d2phi / dx2 using finite difference
d2phi_dx2 = np.zeros_like(phi)
d2phi_dx2[:, 1:-1] = (phi[:, 2:] - 2*phi[:, 1:-1] + phi[:, :-2]) / dx**2

# Compute second derivative d2phi/ d2y using finite difference
d2phi_dy2 = np.zeros_like(phi)
d2phi_dy2[1:-1, :] = (phi[2:, :] - 2*phi[1:-1, :] + phi[:-2, :]) / dy**2

# Laplacian of phi
laplacian_phi = d2phi_dx2 + d2phi_dy2

# Visualize the Laplacian
plt.matshow(laplacian_phi, cmap='coolwarm')
plt.colorbar(label='Laplacian of φ')
plt.title('Laplacian ∇²φ')
plt.xlabel('x')
plt.ylabel('y')
plt.show()