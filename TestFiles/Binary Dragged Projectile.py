import numpy as np
import matplotlib.pyplot as plt

# Constants
v_0 = 100  # Initial velocity (m/s)
angle = 30  # Launch angle (degrees)
theta = np.deg2rad(angle)  # Convert angle to radians
t0, tf, dt = 0, 12, 0.01  # Time parameters
g0 = 9.8

# Time array
t = np.arange(t0, tf, dt)

# Acceleration due to gravity (m/s^2)
a_g = np.array([0, -9.8])

# Initial conditions
v0 = np.array([v_0 * np.cos(theta), v_0 * np.sin(theta)])
r0 = np.array([0, 0])

k = 0.9

def drag_a(t,r,v, k=0.1, m=1):
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return np.array([0, 0])  # Avoid division by zero
    drag_acc = - k * norm_v**2  * (v / norm_v) / m
    return drag_acc + a_g

def kinematic_rk4(t, f, r0, v0):
    dt = t[1] - t[0]  # Assuming uniform time steps
    half_dt = dt / 2
    # Initialize arrays to store positions, velocities, and accelerations
    dim = len(r0)
    r = np.zeros((len(t), dim))  # Position array
    v = np.zeros((len(t), dim))  # Velocity array
    a = np.zeros((len(t), dim))  # Acceleration array
    # Set initial conditions
    r[0] = r0
    v[0] = v0
    a[0] = f(t[0], r0, v0)
    for i in range(len(t) - 1):
        t_i = t[i]
        r_i = r[i]
        v_i = v[i]
        # RK4 coefficients for velocity
        k1_v = f(t_i, r_i, v_i)
        k2_v = f(t_i + half_dt, r_i + v_i * half_dt, v_i + k1_v * half_dt)
        k3_v = f(t_i + half_dt, r_i + v_i * half_dt, v_i + k2_v * half_dt)
        k4_v = f(t_i + dt, r_i + v_i * dt, v_i + k3_v * dt)
        # RK4 coefficients for position
        k1_r = v_i
        k2_r = v_i + k1_v * half_dt
        k3_r = v_i + k2_v * half_dt
        k4_r = v_i + k3_v * dt
        # Update velocity and position
        v[i + 1] = v_i + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * dt / 6
        r[i + 1] = r_i + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) * dt / 6
        # Update acceleration for the next step
        a[i + 1] = f(t[i + 1], r[i + 1], v[i + 1])
    return r, v, a

r,v,a = kinematic_rk4(t,drag_a,r0,v0)

angle_list = np.arange(0,90,5)
for theta in angle_list:
    theta = np.deg2rad(theta)  # Convert angle to radians
    v0 = np.array([v_0 * np.cos(theta), v_0 * np.sin(theta)])
    r,v,a = kinematic_rk4(t,drag_a,r0,v0)
    plt.plot(r[:,0], r[:,1], color = 'r')
    plt.ylim([0,max(r[:,1])*1.2])
    plt.gca().set_aspect('equal')
    plt.grid()
plt.figure(figsize=(10, 6))
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Distance (m)')
plt.title('Projectile Motion with Drag')
plt.gca().set_aspect('equal')
plt.grid(True)
plt.show()

x = np.arange(11,1,dtype=int)
y = x**2
plt.bar(x,y)
plt.grid()
plt.show()