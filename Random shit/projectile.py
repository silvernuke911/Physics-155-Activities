import numpy as np
import matplotlib.pyplot as plt

# Constants
v_0 = 100  # Initial velocity (m/s)
angle = 30  # Launch angle (degrees)
theta = np.deg2rad(angle)  # Convert angle to radians
t0, tf, dt = 0, 12, 0.2  # Time parameters

# Time array
t = np.arange(t0, tf, dt)

# Acceleration due to gravity (m/s^2)
a_g = np.array([0, -9.8])

# Initial conditions
v0 = np.array([v_0 * np.cos(theta), v_0 * np.sin(theta)])
r0 = np.array([0, 0])

def drag_a(v, m=1, k=0.5):
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return np.array([0, 0])  # Avoid division by zero
    drag_acc = -k * norm_v * (v / norm_v) / m
    return drag_acc

def projectile_motion_rk4(t, r0, v0):
    def equations(t, state):
        r, v = state[:2], state[2:]
        a = a_g + drag_a(v)
        return np.concatenate([v, a])
    def rk4_step(f, t, state, dt):
        k1 = dt * f(t, state)
        k2 = dt * f(t + dt / 2, state + k1 / 2)
        k3 = dt * f(t + dt / 2, state + k2 / 2)
        k4 = dt * f(t + dt, state + k3)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    state = np.concatenate([r0, v0])
    r = np.zeros((len(t), 2))  # Position array
    v = np.zeros((len(t), 2))  # Velocity array
    r[0] = r0
    v[0] = v0
    
    for i in range(len(t) - 1):
        state = rk4_step(equations, t[i], state, dt)
        r[i + 1], v[i + 1] = state[:2], state[2:]
    
    return r, v

# Run simulation
r, v = projectile_motion_rk4(t, r0, v0)

# Extract position components
r_x = r[:, 0]
r_y = r[:, 1]

# Plot results
plt.figure(figsize=(10,6))
plt.plot(r_x, r_y, marker = '')
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Distance (m)')
plt.title('Projectile Motion with Drag (RK4)')
plt.grid(True)
plt.show()
