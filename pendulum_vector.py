import numpy as np
import matplotlib.pyplot as plt

# Parameters for the pendulum
g = 9.81  # gravitational acceleration (m/s^2)
L = 1.0   # length of the pendulum (m)

# Define the equations for the pendulum's motion
def dtheta_dt(thetadot):
    """First equation: d(theta)/dt = thetadot."""
    return thetadot

def dthetadot_dt(theta):
    """Second equation: d(thetadot)/dt = -(g/L) * sin(theta)."""
    return -(g / L) * np.sin(theta)

# Generate the grid of (theta, thetadot) values for the phase plot
theta_vals = np.linspace(-2 * np.pi, 2 * np.pi, 100)  # Theta values from -2π to 2π
thetadot_vals = np.linspace(-10, 10, 100)             # Thetadot values from -10 to 10
Theta, Thetadot = np.meshgrid(theta_vals, thetadot_vals)

# Compute the vector field for the phase plot
dTheta = dtheta_dt(Thetadot)  # d(theta)/dt = thetadot
dThetadot = dthetadot_dt(Theta)  # d(thetadot)/dt = -(g/L) * sin(theta)

# Create the stream plot for the phase plot
plt.figure(figsize=(8, 6))
plt.streamplot(Theta, Thetadot, dTheta, dThetadot, color='blue', linewidth=1)

# Add labels and title
plt.xlabel(r'$\theta$ (Angular Position)')
plt.ylabel(r'$\dot{\theta}$ (Angular Velocity)')
plt.title('Phase Plot of a Pendulum')
plt.grid(True)
plt.xlim([-2 * np.pi, 2 * np.pi])  # Limit x-axis to -2π to 2π
plt.ylim([-10, 10])  # Limit y-axis to -10 to 10

# Add the zero-energy equilibrium points (stable at 0, unstable at ±π)
plt.scatter([0, np.pi, -np.pi], [0, 0, 0], color='red', zorder=5, label='Equilibrium Points')
plt.legend()
plt.show()
