import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
from matplotlib.animation import PillowWriter
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.formatter.use_mathtext']=True
mpl.rcParams.update({'font.size': 12})

def rk4_heat_equation(T0, L, alpha, dt, dx, t_end):
    """
    Solve the heat equation using RK4 method with fixed boundary conditions.

    Parameters:
    T0 : numpy.ndarray
        Initial temperature distribution (array of length N).
    L : float
        Length of the rod.
    alpha : float
        Thermal diffusivity.
    dt : float
        Time step size.
    dx : float
        Spatial step size.
    t_end : float
        End time.

    Returns:
    t_values : numpy.ndarray
        The array of time values.
    T_values : numpy.ndarray
        The array of temperature distributions at each time step.
    """
    N = len(T0)        # Number of spatial points
    M = int(t_end / dt)  # Number of time steps

    # Initialize arrays
    t_values = np.linspace(0, t_end, M + 1)
    T_values = np.zeros((M + 1, N))
    T_values[0, :] = T0

    # Define the spatial derivative matrix
    def spatial_derivative(T):
        D = np.zeros((N, N))
        for i in range(1, N - 1):
            D[i, i - 1] = 1
            D[i, i] = -2
            D[i, i + 1] = 1
        D = D / (dx ** 2)
        return D

    # Define the RK4 system
    def rk4_step(T):
        D = spatial_derivative(T)
        k1 = dt * alpha * np.dot(D, T)
        k2 = dt * alpha * np.dot(D, T + 0.5 * k1)
        k3 = dt * alpha * np.dot(D, T + 0.5 * k2)
        k4 = dt * alpha * np.dot(D, T + k3)
        return T + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Perform the RK4 integration
    for i in range(M):
        T_values[i + 1, :] = rk4_step(T_values[i, :])
    
    return t_values, T_values

# Parameters
L = 10.0           # Length of the rod
alpha = 0.5       # Thermal diffusivity
dx = 0.1           # Spatial step size
dt = 0.01          # Time step size
t_end = 1.0        # End time

# Initial temperature distribution
x = np.arange(0, L + dx, dx)
T0 = x**2 # Example initial condition

# Solve the heat equation
t_values, T_values = rk4_heat_equation(T0, L, alpha, dt, dx, t_end)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(0, len(t_values), int(len(t_values)/10)):  # Plot every 10th time step
    plt.plot(x, T_values[i, :], label=f't = {t_values[i]:.2f}s')

plt.xlabel('Position x')
plt.ylabel('Temperature T')
plt.title('Temperature Distribution Over Time')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def rk4_heat_equation_neumann(T0, L, alpha, dt, dx, t_end):
    """
    Solve the heat equation using RK4 method with Neumann boundary conditions (dT/dx = 0 at boundaries).

    Parameters:
    T0 : numpy.ndarray
        Initial temperature distribution (array of length N).
    L : float
        Length of the rod.
    alpha : float
        Thermal diffusivity.
    dt : float
        Time step size.
    dx : float
        Spatial step size.
    t_end : float
        End time.

    Returns:
    t_values : numpy.ndarray
        The array of time values.
    T_values : numpy.ndarray
        The array of temperature distributions at each time step.
    """
    N = len(T0)        # Number of spatial points
    M = int(t_end / dt)  # Number of time steps

    # Initialize arrays
    t_values = np.linspace(0, t_end, M + 1)
    T_values = np.zeros((M + 1, N))
    T_values[0, :] = T0

    # Define the spatial derivative matrix with Neumann boundary conditions
    def spatial_derivative(T):
        D = np.zeros((N, N))
        for i in range(1, N - 1):
            D[i, i - 1] = 1
            D[i, i] = -2
            D[i, i + 1] = 1
        D = D / (dx ** 2)
        
        # Adjust for Neumann boundary conditions
        D[0, 0] = -2 / (dx ** 2)  # Boundary condition at x=0
        D[0, 1] = 2 / (dx ** 2)
        D[-1, -2] = 2 / (dx ** 2)  # Boundary condition at x=L
        D[-1, -1] = -2 / (dx ** 2)
        
        return D

    # Define the RK4 system
    def rk4_step(T):
        D = spatial_derivative(T)
        k1 = dt * alpha * np.dot(D, T)
        k2 = dt * alpha * np.dot(D, T + 0.5 * k1)
        k3 = dt * alpha * np.dot(D, T + 0.5 * k2)
        k4 = dt * alpha * np.dot(D, T + k3)
        return T + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Perform the RK4 integration
    for i in range(M):
        T_values[i + 1, :] = rk4_step(T_values[i, :])
    
    return t_values, T_values

# Parameters
L = 10.0           # Length of the rod
alpha = 0.6       # Thermal diffusivity
dx = 0.1           # Spatial step size
dt = 0.01          # Time step size
t_end = 10.0        # End time

# Initial temperature distribution
x = np.arange(0, L + dx, dx)
T0 = np.sin(np.pi * x / L)  # Example initial condition

# Solve the heat equation
t_values, T_values = rk4_heat_equation_neumann(T0, L, alpha, dt, dx, t_end)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(0, len(t_values), int(len(t_values)/20)):  # Plot every 10th time step
    plt.plot(x, T_values[i, :], label=f't = {t_values[i]:.2f}s')

plt.xlabel(r'Position $x$')
plt.ylabel(r'Temperature $T$')
plt.title('Temperature Distribution Over Time with Neumann Boundary Conditions')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def rk4_heat_equation_2d_neumann(T0, Lx, Ly, alpha, dt, dx, dy, t_end):
    """
    Solve the 2D heat equation using RK4 method with Neumann boundary conditions (dT/dx = 0 and dT/dy = 0 at boundaries).

    Parameters:
    T0 : numpy.ndarray
        Initial temperature distribution (2D array).
    Lx : float
        Length of the rod in x-direction.
    Ly : float
        Length of the rod in y-direction.
    alpha : float
        Thermal diffusivity.
    dt : float
        Time step size.
    dx : float
        Spatial step size in x-direction.
    dy : float
        Spatial step size in y-direction.
    t_end : float
        End time.

    Returns:
    t_values : numpy.ndarray
        The array of time values.
    T_values : numpy.ndarray
        The array of temperature distributions at each time step (3D array).
    """
    Nx = len(T0)        # Number of spatial points in x-direction
    Ny = len(T0[0])     # Number of spatial points in y-direction
    M = int(t_end / dt)  # Number of time steps

    # Initialize arrays
    t_values = np.linspace(0, t_end, M + 1)
    T_values = np.zeros((M + 1, Nx, Ny))
    T_values[0, :, :] = T0

    # Define the spatial derivative matrix for 2D
    def spatial_derivative_2d(T):
        D_x = np.zeros((Nx, Nx))
        D_y = np.zeros((Ny, Ny))
        
        # Construct x-direction derivative matrix
        for i in range(1, Nx - 1):
            D_x[i, i - 1] = 1
            D_x[i, i] = -2
            D_x[i, i + 1] = 1
        D_x = D_x / (dx ** 2)
        
        # Construct y-direction derivative matrix
        for j in range(1, Ny - 1):
            D_y[j, j - 1] = 1
            D_y[j, j] = -2
            D_y[j, j + 1] = 1
        D_y = D_y / (dy ** 2)

        # Adjust for Neumann boundary conditions
        D_x[0, 0] = -2 / (dx ** 2)  # Boundary condition at x=0
        D_x[0, 1] = 2 / (dx ** 2)
        D_x[-1, -2] = 2 / (dx ** 2)  # Boundary condition at x=Lx
        D_x[-1, -1] = -2 / (dx ** 2)
        
        D_y[0, 0] = -2 / (dy ** 2)  # Boundary condition at y=0
        D_y[0, 1] = 2 / (dy ** 2)
        D_y[-1, -2] = 2 / (dy ** 2)  # Boundary condition at y=Ly
        D_y[-1, -1] = -2 / (dy ** 2)
        
        # Create 2D Laplacian
        D = np.zeros((Nx, Ny))
        for i in range(Nx):
            for j in range(Ny):
                D[i, j] = np.dot(D_x[i, :], T[i, :]) + np.dot(D_y[j, :], T[:, j])
        
        return D

    # Define the RK4 system
    def rk4_step_2d(T):
        D = spatial_derivative_2d(T)
        k1 = dt * alpha * D
        k2 = dt * alpha * spatial_derivative_2d(T + 0.5 * k1)
        k3 = dt * alpha * spatial_derivative_2d(T + 0.5 * k2)
        k4 = dt * alpha * spatial_derivative_2d(T + k3)
        return T + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Perform the RK4 integration
    for i in range(M):
        T_values[i + 1, :, :] = rk4_step_2d(T_values[i, :, :])
    
    return t_values, T_values

# Parameters
Lx = 10.0           # Length of the rod in x-direction
Ly = 10.0           # Length of the rod in y-direction
alpha = 0.01        # Thermal diffusivity
dx = 0.5            # Spatial step size in x-direction
dy = 0.5            # Spatial step size in y-direction
dt = 0.01           # Time step size
t_end = 1.0         # End time

# Initial temperature distribution
x = np.arange(0, Lx + dx, dx)
y = np.arange(0, Ly + dy, dy)
X, Y = np.meshgrid(x, y)
T0 = np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)  # Example initial condition

# Solve the heat equation
t_values, T_values = rk4_heat_equation_2d_neumann(T0, Lx, Ly, alpha, dt, dx, dy, t_end)

# Plot the results
plt.figure(figsize=(12, 6))
plt.imshow(T_values[-1, :, :], extent=[0, Lx, 0, Ly], origin='lower', aspect='auto')
plt.colorbar(label='Temperature T')
plt.title('Temperature Distribution at t = {:.2f}s'.format(t_values[-1]))
plt.xlabel('x')
plt.ylabel('y')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

def rk4_heat_equation_2d_neumann(T0, Lx, Ly, alpha, dt, dx, dy, t_end):
    """
    Solve the 2D heat equation using RK4 method with Neumann boundary conditions (dT/dx = 0 and dT/dy = 0 at boundaries).

    Parameters:
    T0 : numpy.ndarray
        Initial temperature distribution (2D array).
    Lx : float
        Length of the rod in x-direction.
    Ly : float
        Length of the rod in y-direction.
    alpha : float
        Thermal diffusivity.
    dt : float
        Time step size.
    dx : float
        Spatial step size in x-direction.
    dy : float
        Spatial step size in y-direction.
    t_end : float
        End time.

    Returns:
    t_values : numpy.ndarray
        The array of time values.
    T_values : numpy.ndarray
        The array of temperature distributions at each time step (3D array).
    """
    Nx = len(T0)        # Number of spatial points in x-direction
    Ny = len(T0[0])     # Number of spatial points in y-direction
    M = int(t_end / dt)  # Number of time steps

    # Initialize arrays
    t_values = np.linspace(0, t_end, M + 1)
    T_values = np.zeros((M + 1, Nx, Ny))
    T_values[0, :, :] = T0

    # Define the spatial derivative matrix for 2D
    def spatial_derivative_2d(T):
        D_x = np.zeros((Nx, Nx))
        D_y = np.zeros((Ny, Ny))
        
        # Construct x-direction derivative matrix
        for i in range(1, Nx - 1):
            D_x[i, i - 1] = 1
            D_x[i, i] = -2
            D_x[i, i + 1] = 1
        D_x = D_x / (dx ** 2)
        
        # Construct y-direction derivative matrix
        for j in range(1, Ny - 1):
            D_y[j, j - 1] = 1
            D_y[j, j] = -2
            D_y[j, j + 1] = 1
        D_y = D_y / (dy ** 2)

        # Adjust for Neumann boundary conditions
        D_x[0, 0] = -2 / (dx ** 2)  # Boundary condition at x=0
        D_x[0, 1] = 2 / (dx ** 2)
        D_x[-1, -2] = 2 / (dx ** 2)  # Boundary condition at x=Lx
        D_x[-1, -1] = -2 / (dx ** 2)
        
        D_y[0, 0] = -2 / (dy ** 2)  # Boundary condition at y=0
        D_y[0, 1] = 2 / (dy ** 2)
        D_y[-1, -2] = 2 / (dy ** 2)  # Boundary condition at y=Ly
        D_y[-1, -1] = -2 / (dy ** 2)
        
        # Create 2D Laplacian
        D = np.zeros((Nx, Ny))
        for i in range(Nx):
            for j in range(Ny):
                D[i, j] = np.dot(D_x[i, :], T[i, :]) + np.dot(D_y[j, :], T[:, j])
        
        return D

    # Define the RK4 system
    def rk4_step_2d(T):
        D = spatial_derivative_2d(T)
        k1 = dt * alpha * D
        k2 = dt * alpha * spatial_derivative_2d(T + 0.5 * k1)
        k3 = dt * alpha * spatial_derivative_2d(T + 0.5 * k2)
        k4 = dt * alpha * spatial_derivative_2d(T + k3)
        return T + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Perform the RK4 integration
    for i in range(M):
        T_values[i + 1, :, :] = rk4_step_2d(T_values[i, :, :])
    
    return t_values, T_values

def save_gif(T_values, t_values, filename):
    """
    Save the temperature distributions as a GIF.

    Parameters:
    T_values : numpy.ndarray
        The array of temperature distributions at each time step (3D array).
    t_values : numpy.ndarray
        The array of time values.
    filename : str
        The name of the GIF file to save.
    """
    fig, ax = plt.subplots()
    ims = []

    for i in range(len(t_values)):
        im = ax.imshow(T_values[i, :, :], animated=True, cmap='hot', interpolation='bilinear')
        ims.append([im])
        print(f'done frame {i}')

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    ani.save(filename, writer='pillow')
    

# Parameters
Lx = 10.0           # Length of the rod in x-direction
Ly = 10.0           # Length of the rod in y-direction
alpha = 0.01        # Thermal diffusivity
dx = 0.5            # Spatial step size in x-direction
dy = 0.5            # Spatial step size in y-direction
dt = 0.01           # Time step size
t_end = 1.0         # End time

# Initial temperature distribution
x = np.arange(0, Lx + dx, dx)
y = np.arange(0, Ly + dy, dy)
X, Y = np.meshgrid(x, y)
T0 = np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)  # Example initial condition

# Solve the heat equation
t_values, T_values = rk4_heat_equation_2d_neumann(T0, Lx, Ly, alpha, dt, dx, dy, t_end)

# Save animation as GIF
save_gif(T_values, t_values, 'heat_equation_animation.gif')

import os
print("Current working directory:", os.getcwd())