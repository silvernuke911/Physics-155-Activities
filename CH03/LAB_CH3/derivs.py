import numpy as np
import matplotlib.pyplot as plt

def latex_font():
    import matplotlib as mpl
    import matplotlib.font_manager as font_manager
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.unicode_minus': False,
        'axes.formatter.use_mathtext': True,
        'font.size': 12
    })
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
    
def f_func(x):
    # Express the function above. Function takes 
    # in a NumPy array and returns another NumPy
    # array of the same size. 
    return 2 + 5 * np.sin(x) + 0.1 * np.sin(30 * x)

def g_func(x):
    # Same as f(x) but remove the high-frequency
    # oscillation (the "noise").
    return 2 + 5 * np.sin(x)

def f_adiv(x):
    # Express the derivative of f(x) above obtained
    # analytically.
    return 5 * np.cos(x) + (0.1 / 30) * np.cos(30 * x) 

def forward_diff(func, x, h):
    # Calculate the derivative using the forward
    # difference method. Part of the input is the
    # function `func` of which the derivative is
    # taken from.
    output = np.zeros_like(x)
    for i in range(len(x)):
        if i == len(x)-1:
            output[i] = (func(x[i]) - func(x[i-1])) / h
        elif i < len(x)-1:
            output[i] = (func(x[i+1]) - func(x[i])) / h
    return output

def central_diff(func,x,h):
    output = np.zeros_like(x)
    for i in range(len(x)):
        if i==0:
            output[i] = (func(x[i+1]) - func(x[i])) / h
        elif i==len(x)-1:
            output[i] = (func(x[i]) - func(x[i-1])) / h 
        else:
            output[i] = (func(x[i+1]) - func(x[i-1])) / (2 * h)
    return output

## You may copy this part
fig, ax = plt.subplots()

h = 0.1 # our baseline h
# Create equally-spaced array with distance h
x = np.arange(-2*np.pi, 2*np.pi, h)
# Plot the baselines
ax.plot(x, f_adiv(x), color="k", label="Analytical derivative")
ax.plot(x, central_diff(g_func, x, h), color='r', label="Noiseless")
# Plot the numerical derivatives using different h
for hv in [h/2, h, 2*h]:
    x = np.arange(-2*np.pi, 2*np.pi, hv)
    ax.plot(x, central_diff(f_func, x, hv), label=f"Central diff, h={hv}")
    
ax.grid()
ax.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def latex_font():
    import matplotlib as mpl
    import matplotlib.font_manager as font_manager
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.unicode_minus': False,
        'axes.formatter.use_mathtext': True,
        'font.size': 12
    })
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
latex_font()
def f_func(x):
    # Express the function above. Function takes 
    # in a NumPy array and returns another NumPy
    # array of the same size. 
    return 2 + 5 * np.sin(x) + 0.1 * np.sin(30 * x)

def g_func(x):
    # Same as f(x) but remove the high-frequency
    # oscillation (the "noise").
    return 2 + 5 * np.sin(x)

def f_adiv(x):
    # Express the derivative of f(x) above obtained
    # analytically.
    return 5 * np.cos(x) + (0.1 / 30) * np.cos(30 * x) 

def forward_diff(func, x, h):
    # Calculate the derivative using the forward
    # difference method. Part of the input is the
    # function `func` of which the derivative is
    # taken from.
    output = np.zeros_like(x)
    for i in range(len(x)):
        if i == len(x)-1:
            output[i] = (func(x[i]) - func(x[i-1])) / h
        elif i < len(x)-1:
            output[i] = (func(x[i+1]) - func(x[i])) / h
    return output

def central_diff(func,x,h):
    output = np.zeros_like(x)
    for i in range(len(x)):
        if i==0:
            output[i] = (func(x[i+1]) - func(x[i])) / h
        elif i==len(x)-1:
            output[i] = (func(x[i]) - func(x[i-1])) / h 
        else:
            output[i] = (func(x[i+1]) - func(x[i-1])) / (2 * h)
    return output

## You may copy this part
fig, ax = plt.subplots()

h = 0.1 # our baseline h
# Create equally-spaced array with distance h
x = np.arange(-2*np.pi, 2*np.pi, h)
# Plot the baselines
# ax.plot(x,f_func(x), color = 'b', label = 'Orig function')
ax.plot(x, f_adiv(x), color="k", label="Analytical derivative")
ax.plot(x, central_diff(g_func, x, h), color='r', label="Noiseless")
# Plot the numerical derivatives using different h
for hv in [h/2, h, 2*h]:
    x = np.arange(-2*np.pi, 2*np.pi, hv)
    ax.plot(x, central_diff(f_func, x, hv), label=f"Central diff, h={hv}")
ax.grid()
ax.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x_grid = np.linspace(-1, 3, 100)
y_grid = np.linspace(-2, 1.5, 100)

# Original function
X, Y = np.meshgrid(x_grid, y_grid)
phi = X**2 - 2*X + Y**4 - 2*Y**2 + Y

ms = ax.matshow(phi, vmin=-3, vmax=0.6, origin="lower", alpha=0.7, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], cmap = 'inferno')
cp = ax.contour(X, Y, phi, levels=[-3, -2.4, -1.8, -1, -0.5, 0.6])

bgcb = fig.colorbar(ms, ax=ax)
fig.colorbar(cp, cax=bgcb.ax)
plt.title(r'$\phi(x)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

dx = np.diff(x_grid)[0]
dy = np.diff(y_grid)[0]

# Compute partial derivative dphi/dx and dphi/dy using forward difference (np.roll)
dphi_dx = (np.roll(phi, -1, axis=1) - phi) / dx
dphi_dy = (np.roll(phi, -1, axis=0) - phi) / dy

# Removing edges
dphi_dx = dphi_dx[1:-1,1:-1]
dphi_dy = dphi_dy[1:-1,1:-1]

# Plotting the partial derivatives
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
extent = [x_grid[1], x_grid[-2], y_grid[1], y_grid[-2]]

# Plot for dphi / dx
c1 = ax[0].contourf(dphi_dx, cmap="inferno", levels=100, extent = extent)
fig.colorbar(c1, ax=ax[0])
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$y$')
ax[0].set_title(r"Partial Derivative $\frac{\partial \phi}{\partial x}$")

# Plot for dphi / dy
c2 = ax[1].contourf(dphi_dy, cmap="inferno", levels=100, extent = extent)
fig.colorbar(c2, ax=ax[1])
ax[1].set_xlabel('$x$')
ax[1].set_ylabel('$y$')
ax[1].set_title(r"Partial Derivative $\frac{\partial \phi}{\partial y}$")

plt.show()

# Compute second derivative d2phi / dx2 and d2phi / dy2   using finite difference
d2phi_dx2 = np.zeros_like(phi)
d2phi_dx2[:, 1:-1] = (phi[:, 2:] - 2*phi[:, 1:-1] + phi[:, :-2]) / dx**2

d2phi_dy2 = np.zeros_like(phi)
d2phi_dy2[1:-1, :] = (phi[2:, :] - 2*phi[1:-1, :] + phi[:-2, :]) / dy**2

# Testing thingies
    # plt.matshow(d2phi_dx2, cmap='inferno')
    # plt.colorbar()
    # plt.matshow(d2phi_dy2, cmap='inferno')
    # plt.colorbar()

# Laplacian of phi and removal of edges
laplacian_phi = d2phi_dx2 + d2phi_dy2
laplacian_phi = laplacian_phi[1:-1, 1:-1]

extent = [x_grid[1], x_grid[-2], y_grid[1], y_grid[-2]]
# Visualize the Laplacian
plt.contourf(laplacian_phi, cmap='inferno', levels = 100, extent = extent)
plt.colorbar(label=r'$\nabla^2 \phi$')
plt.title(r'Laplacian $\nabla^2 \phi$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

# Compute the partial derivatives using central difference
dphi_dx = np.zeros_like(phi)
dphi_dx[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * dx)

dphi_dy = np.zeros_like(phi)
dphi_dy[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * dy)

# Downsample the grid to reduce the number of arrows
step = 3 
X_down = X[1:-1:step, 1:-1:step]
Y_down = Y[1:-1:step, 1:-1:step]
dphi_dx_down = dphi_dx[1:-1:step, 1:-1:step]
dphi_dy_down = dphi_dy[1:-1:step, 1:-1:step]

# Make a quiver plot
plt.figure(figsize=(6, 6))
plt.quiver(X_down, Y_down, dphi_dx_down, dphi_dy_down, color='r', scale=15, scale_units='inches')
plt.title(r'$\nabla\phi(x, y)$')

# Fix the axis limits
plt.xlim([np.min(x_grid), np.max(x_grid)])
plt.ylim([np.min(y_grid), np.max(y_grid)])

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.show()