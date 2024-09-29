import numpy as np
import matplotlib.pyplot as plt
import time

def compute_orbital_elements(r_i, v_i, G, M):
    if len(r_i) == 2:  # Create 3D vectors if input is 2D
        r_i = np.append(r_i, 0)
        v_i = np.append(v_i, 0)
    mu = G * M  # Gravitational parameter
    h = np.cross(r_i, v_i)
    e = (np.cross(v_i, h) / mu) - (r_i / np.linalg.norm(r_i))
    e_mag = np.linalg.norm(e)
    a = 1 / ((2 / np.linalg.norm(r_i)) - (np.linalg.norm(v_i) ** 2 / mu))
    T_p = 2 * np.pi * np.sqrt(a**3 / mu)
    return {
        'semi_major_axis': a,
        'eccentricity': e_mag,
        'orbital_period': T_p
    }

def kinematic_rk4(t, f, r0, v0):
    dt = t[1] - t[0]  # Time step
    r = np.zeros((len(t), len(r0)))  # Position array
    v = np.zeros((len(t), len(v0)))  # Velocity array
    a = np.zeros((len(t), len(v0)))  # Acceleration array
    r[0], v[0] = r0, v0
    a[0] = f(t[0], r0, v0)
    for i in range(len(t) - 1):
        k1_v = f(t[i], r[i], v[i])
        k2_v = f(t[i] + dt / 2, r[i] + v[i] * dt / 2, v[i] + k1_v * dt / 2)
        k3_v = f(t[i] + dt / 2, r[i] + v[i] * dt / 2, v[i] + k2_v * dt / 2)
        k4_v = f(t[i] + dt, r[i] + v[i] * dt, v[i] + k3_v * dt)
        
        k1_r = v[i]
        k2_r = v[i] + k1_v * dt / 2
        k3_r = v[i] + k2_v * dt / 2
        k4_r = v[i] + k3_v * dt
        
        v[i + 1] = v[i] + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * dt / 6
        r[i + 1] = r[i] + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) * dt / 6
        a[i + 1] = f(t[i + 1], r[i + 1], v[i + 1])
    return r, v, a

def euler_grav(t, f, r0, v0):
    dt = t[1] - t[0]
    r = np.zeros((len(t), len(r0)))
    v = np.zeros((len(t), len(v0)))
    a = np.zeros((len(t), len(v0)))
    r[0], v[0] = r0, v0
    a[0] = f(t[0], r0, v0)
    for i in range(1, len(t)):
        v[i] = v[i - 1] + a[i - 1] * dt
        r[i] = r[i - 1] + v[i - 1] * dt
        a[i] = f(t[i], r[i], v[i])
    return r, v, a

def grav_acc(t, r, v, m=1):
    G = 1
    mu = G * m
    r_mag = np.linalg.norm(r)
    r_norm = lambda r, r_mag: r / r_mag if r_mag != 0 else r_mag * 0
    return -(mu / r_mag**2) * r_norm(r, r_mag)  #0.01 * v

# Energy calculations
def kinetic_energy(v, m=1):
    return 0.5 * m * np.sum(v**2, axis=1)

def potential_energy(r, m=1):
    G = 1
    r_mag = np.linalg.norm(r, axis=1)
    return -G * m**2 / r_mag

def total_energy(r, v, m=1):
    return kinetic_energy(v, m) + potential_energy(r, m)

def plot_orbits():
    print("\n Plotting Orbits")
    plt.plot(r_rk4[:, 0], r_rk4[:, 1], color='blue', label='RK4')
    plt.plot(r_elr[:, 0], r_elr[:, 1], color='red', label='Euler')
    plt.scatter(0, 0, marker='o', s=80, color='yellow', label='Central Body')
    plt.scatter(r_i[0],r_i[1],marker='.',s=60,color='red',zorder=2)
    plt.xlabel('$x$-axis (AU)')
    plt.ylabel('$y$-axis (AU)')
    plt.title('Planetary Orbit Simulation')
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_energy():
    print("\n Plotting Energy")
    t_plot = t[:-1]
    plt.plot(t_plot, TME_rk4[:-1], color='blue', label='RK4 Total Energy')
    plt.plot(t_plot, TME_elr[:-1], color='red', label='Euler Total Energy')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Conservation Comparison')
    plt.legend()
    plt.grid()
    plt.show()

def plot_errors():
    print("\n Plotting Errors")
    pos_error = np.linalg.norm(r_rk4 - r_elr, axis=1)
    plt.plot(t, pos_error, label='Position Error: RK4 vs Euler', color='green')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title('Position Error Over Time')
    plt.legend()
    plt.grid()
    plt.show()

# Set initial conditions
r_i = np.array([1, 0])
v_i = np.array([0, 1])
dt = 0.0001
t = np.arange(0, 40 + dt, dt)

# Run simulations
r_rk4, v_rk4, a_rk4 = kinematic_rk4(t, grav_acc, r_i, v_i)
r_elr, v_elr, a_elr = euler_grav(t, grav_acc, r_i, v_i)

# Calculate total energy for both methods
TME_rk4 = total_energy(r_rk4, v_rk4)
TME_elr = total_energy(r_elr, v_elr)

# Plotting results
plot_orbits()
plot_energy()
plot_errors()
