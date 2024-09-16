import numpy as np

def rk4(f, t0, y0, t_f, h):
    """
    Solve the ODE dy/dt = f(t, y) using the RK4 method.

    Parameters:
    f : function
        The function defining the differential equation dy/dt = f(t, y).
    t0 : float
        The initial time.
    y0 : float
        The initial value of the function.
    t_f : float
        The end time for the integration.
    h : float
        The step size.

    Returns:
    t_values : numpy.ndarray
        The array of time values.
    y_values : numpy.ndarray
        The array of solution values at each time step.
    """
    # Number of steps
    n_steps = int((t_f - t0) / h)
    
    # Initialize arrays to store the time values and the solution values
    t_values = np.linspace(t0, t_f, n_steps + 1)
    y_values = np.zeros(n_steps + 1)
    
    # Set the initial values
    t_values[0] = t0
    y_values[0] = y0
    
    # Perform the RK4 integration
    for i in range(n_steps):
        t = t_values[i]
        y = y_values[i]
        
        # Compute k1, k2, k3, k4
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)
        
        # Update y
        y_values[i + 1] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return t_values, y_values

# Example differential equation: dy/dt = y - t^2 + 1
k = 0.5
def f(t, y):
    return - k * y

# Example usage
t0 = 0       # Initial time
y0 = 0.5     # Initial value
t_f = 10    # End time
h = 0.2      # Step size

t_values, y_values = rk4(f, t0, y0, t_f, h)

# Display results
import matplotlib.pyplot as plt

plt.plot(t_values, y_values, label='RK4 Solution')
plt.xlabel('t')
plt.ylabel('y')
plt.title('RK4 Method Solution')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def rk4_system(f, y0, v0, t0, t_end, h):
    """
    Solve a system of ODEs using the RK4 method.

    Parameters:
    f : function
        The function defining the system of ODEs.
    y0 : float
        The initial value of y.
    v0 : float
        The initial value of v.
    t0 : float
        The initial time.
    t_end : float
        The end time for the integration.
    h : float
        The step size.

    Returns:
    t_values : numpy.ndarray
        The array of time values.
    y_values : numpy.ndarray
        The array of y values at each time step.
    v_values : numpy.ndarray
        The array of v values at each time step.
    """
    # Number of steps
    n_steps = int((t_end - t0) / h)
    
    # Initialize arrays to store the time values and the solution values
    t_values = np.linspace(t0, t_end, n_steps + 1)
    y_values = np.zeros(n_steps + 1)
    v_values = np.zeros(n_steps + 1)
    
    # Set the initial values
    t_values[0] = t0
    y_values[0] = y0
    v_values[0] = v0
    
    # Define the system of ODEs
    def system(t, state):
        y, v = state
        return np.array([v, -omega**2 * y])
    
    # Perform the RK4 integration
    for i in range(n_steps):
        t = t_values[i]
        state = np.array([y_values[i], v_values[i]])
        
        # Compute k1, k2, k3, k4
        k1 = h * system(t, state)
        k2 = h * system(t + h / 2, state + k1 / 2)
        k3 = h * system(t + h / 2, state + k2 / 2)
        k4 = h * system(t + h, state + k3)
        
        # Update y and v
        state += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y_values[i + 1], v_values[i + 1] = state
    
    return t_values, y_values, v_values

# Parameters for the harmonic oscillator
omega = 2 * np.pi  # Angular frequency (e.g., 1 Hz frequency)
y0 = 1            # Initial position
v0 = 0            # Initial velocity
t0 = 0            # Initial time
t_end = 10        # End time
h = 0.01           # Step size

# Solve the system
t_values, y_values, v_values = rk4_system(lambda t, state: np.array([state[1], -omega**2 * state[0]]), y0, v0, t0, t_end, h)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(t_values, y_values, label='Position y(t)')
plt.xlabel('Time t')
plt.ylabel('y(t)')
plt.title('Position vs Time')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_values, v_values, label='Velocity v(t)', color='r')
plt.xlabel('Time t')
plt.ylabel('v(t)')
plt.title('Velocity vs Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def rk4_orbit(f, r0, v0, t0, t_end, h):
    """
    Solve the orbital motion using the RK4 method.

    Parameters:
    f : function
        The function defining the system of ODEs.
    r0 : numpy.ndarray
        The initial position vector (x0, y0).
    v0 : numpy.ndarray
        The initial velocity vector (vx0, vy0).
    t0 : float
        The initial time.
    t_end : float
        The end time for the integration.
    h : float
        The step size.

    Returns:
    t_values : numpy.ndarray
        The array of time values.
    r_values : numpy.ndarray
        The array of position vectors at each time step.
    v_values : numpy.ndarray
        The array of velocity vectors at each time step.
    """
    # Number of steps
    n_steps = int((t_end - t0) / h)
    
    # Initialize arrays to store the time values and the solution values
    t_values = np.linspace(t0, t_end, n_steps + 1)
    r_values = np.zeros((n_steps + 1, 2))
    v_values = np.zeros((n_steps + 1, 2))
    
    # Set the initial values
    t_values[0] = t0
    r_values[0] = r0
    v_values[0] = v0
    
    # Define the system of ODEs
    def system(t, state):
        r = state[:2]
        v = state[2:]
        x, y = r
        vx, vy = v
        r_norm = np.sqrt(x**2 + y**2)
        a = -G * M / r_norm**3
        return np.array([vx, vy, a * x, a * y])
    
    # Perform the RK4 integration
    for i in range(n_steps):
        t = t_values[i]
        state = np.hstack([r_values[i], v_values[i]])
        
        # Compute k1, k2, k3, k4
        k1 = h * system(t, state)
        k2 = h * system(t + h / 2, state + k1 / 2)
        k3 = h * system(t + h / 2, state + k2 / 2)
        k4 = h * system(t + h, state + k3)
        
        # Update r and v
        state += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        r_values[i + 1] = state[:2]
        v_values[i + 1] = state[2:]
    
    return t_values, r_values, v_values

# Parameters for the orbit
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M = 1.989e30    # Mass of the central body (kg), e.g., Sun's mass
r0 = np.array([1.496e11, 0])  # Initial position vector (m), e.g., distance of Earth from Sun
v0 = np.array([0, 29780])    # Initial velocity vector (m/s), e.g., Earth's orbital velocity
t0 = 0            # Initial time
t_end = 365*24*3600  # End time (1 year, seconds)
h = 24*3600      # Step size (1 day, seconds)

# Solve the system
t_values, r_values, v_values = rk4_orbit(lambda t, state: np.array([state[2], state[3], -G*M*state[0] / np.linalg.norm(state[:2])**3, -G*M*state[1] / np.linalg.norm(state[:2])**3]), r0, v0, t0, t_end, h)

# Plot the results
plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
plt.plot(r_values[:, 0], r_values[:, 1])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Orbit Path')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_values, np.sqrt(r_values[:, 0]**2 + r_values[:, 1]**2))
plt.xlabel('Time (s)')
plt.ylabel('Distance from Origin (m)')
plt.title('Distance Over Time')
plt.grid(True)

plt.tight_layout()
plt.show()

