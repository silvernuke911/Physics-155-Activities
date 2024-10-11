import numpy as np
import matplotlib.pyplot as plt

def leapfrog_orbit(G, M, r0, v0, t0, t_end, h):
    """
    Solve the orbital motion using the Leapfrog method.

    Parameters:
    G : float
        Gravitational constant (m^3 kg^-1 s^-2).
    M : float
        Mass of the central body (kg).
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
    
    # Calculate initial acceleration
    r_norm = np.linalg.norm(r0)
    a = -G * M / r_norm**3 * r0

    # Initialize half-step velocity
    v_half = v0 + 0.5 * a * h

    # Perform the Leapfrog integration
    for i in range(n_steps):
        t = t_values[i]
        
        # Update position
        r_values[i + 1] = r_values[i] + v_half * h
        
        # Compute new acceleration
        r_norm = np.linalg.norm(r_values[i + 1])
        a_new = -G * M / r_norm**3 * r_values[i + 1]
        
        # Update velocity
        v_values[i + 1] = v_half + 0.5 * a_new * h
        
        # Prepare for the next iteration
        v_half = v_values[i + 1]
    
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
t_values, r_values, v_values = leapfrog_orbit(G, M, r0, v0, t0, t_end, h)

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