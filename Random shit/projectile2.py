import numpy as np
import matplotlib.pyplot as plt

# Constants
v_0 = 100  # Initial velocity (m/s)
angle = 30  # Launch angle (degrees)
theta = np.deg2rad(angle)  # Convert angle to radians
t0, tf, dt = 0, 12, 0.05  # Time parameters
g0 = 9.8

# Time array
t = np.arange(t0, tf, dt)

# Acceleration due to gravity (m/s^2)
a_g = np.array([0, -9.8])

# Initial conditions
v0 = np.array([v_0 * np.cos(theta), v_0 * np.sin(theta)])
r0 = np.array([0, 0])

k = 0.9

def drag_a(v, k, m=1):
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return np.array([0, 0])  # Avoid division by zero
    drag_acc = - k * norm_v * (v / norm_v) / m
    return drag_acc

def projectile_motion_verlet(t, r0, v0):
    r = np.zeros((len(t), 2))  # Position array
    v = np.zeros((len(t), 2))  # Velocity array
    a = np.zeros((len(t), 2))  # Acceleration array

    # Initial conditions
    r[0] = r0
    v[0] = v0
    a[0] = a_g + drag_a(v0, k)

    for i in range(len(t) - 1):
        # Update position
        r[i + 1] = r[i] + v[i] * dt + 0.5 * a[i] * dt**2

        # Update acceleration
        a[i + 1] = a_g + drag_a(v[i],k)

        # Update velocity
        v[i + 1] = v[i] + 0.5 * (a[i] + a[i + 1]) * dt

    return r, v, a

def projectile_motion_euler(t, r0, v0):

    r = np.zeros((len(t), 2))  # Position array
    v = np.zeros((len(t), 2))  # Velocity array
    a = np.zeros((len(t), 2))  # Acceleration array

    a[0] = a_g + drag_a(v0, k)
    r[0] = r0
    v[0] = v0
    print(len(t))
    for i in range(len(t) - 1):
        a[i + 1] = a_g + drag_a(v[i],k)
        v[i + 1] = v[i] + a[i] * dt
        r[i + 1] = r[i] + v[i] * dt

    return r, v, a

def projectile_motion_rk4(t, r0, v0):
    dt = t[1] - t[0]  # Assuming uniform time steps

    # Initialize arrays to store positions, velocities, and accelerations
    r = np.zeros((len(t), 2))  # Position array
    v = np.zeros((len(t), 2))  # Velocity array
    a = np.zeros((len(t), 2))  # Acceleration array

    # Set initial conditions
    r[0] = r0
    v[0] = v0
    a[0] = a_g + drag_a(v0, k)

    def acc_func(v):
        return a_g + drag_a(v,k)
    
    for i in range(len(t) - 1):
        # Current state
        r_i = r[i]
        v_i = v[i]
        a_i = a[i]
        
        # RK4 coefficients
        k1_v = acc_func(v_i)
        k1_r = v_i
        k2_v = acc_func(v_i + k1_v * dt/2)
        k2_r = v_i + 0.5 * k1_v * dt
        k3_v = acc_func(v_i + k2_v * dt/2)
        k3_r = v_i + 0.5 * k2_v * dt
        k4_v = acc_func(v_i + k3_v * dt)
        k4_r = v_i + k3_v * dt

        # Update values using RK4
        v[i + 1] = v_i + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * dt / 6
        r[i + 1] = r_i + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) * dt / 6
        
        # Compute acceleration for the next step
        a[i + 1] = a_g + drag_a(v[i + 1],k)
    
    return r, v, a

def projectile_motion_actual(t, r0, v0):
    r0_x = r0[0]
    r0_y = r0[1]
    v0_x = v0[0]
    v0_y = v0[1]
    a = np.zeros_like(t)
    def horizontal_motion(t,r0,v0):
        v = 1 / ((1 / v0) + (k * t))
        r = r0 + (1/k)*np.log(1 + k* v0 * t)
        return r,v
    
    def vertical_motion(t,r0,v0):
        # check velocities
        tau = np.sqrt(g0/k)
        t_crit = (tau / g0) * np.arctan(v0/tau)
        r = np.zeros_like(t)
        v = np.zeros_like(t)
        for i,t_val in enumerate(t):
            if t_val<= t_crit:
                v[i] = tau * np.tan(np.arctan(v0/tau) - (g0/tau)*t_val)
                r[i] = r0 + (tau*2/g0)*np.log((np.cos(np.arctan(v0/tau) - (g0/tau)*t_val))/(np.cos(np.arctan(v0/tau))))
            elif t_val > t_crit:
                v0 = 0
                r0 = np.max(r)
                v[i] = tau * np.tanh(np.arctanh(v0/tau) - (g0/tau) * t_val)
                r[i] = r0 - (tau*2/g0)*np.log((np.cosh(np.arctanh(v0/tau) + (g0/tau)*(t_val-t_crit)))/(np.cosh(np.arctanh(v0/tau))))
        return r,v,a
    print(r0,v0)
    print(r0_x, v0_x)
    horiz = horizontal_motion(t,r0_x,v0_x)
    verti = vertical_motion(t,r0_y,v0_y)
    r_x,r_y = horiz[0],verti[0]
    v_x,v_y = horiz[1],verti[1]

    plt.plot(t,r_x)
    plt.plot(t,r_y)
    plt.plot(r_x,r_y)
    plt.show()

    r_total = np.array([r_x,r_y])
    v_total = np.array([v_x,v_y])

    return r_total,v_total, a

# Run simulation
r_vrl, v_vrl, a_vrl = projectile_motion_verlet(t, r0, v0)
r_elr, v_elr, a_elr = projectile_motion_euler(t, r0, v0)
r_rk4, v_rk4, a_rk4 = projectile_motion_rk4(t, r0, v0)

dt = 0.0001
t_precise = np.arange(t0,tf,dt)
r_act, v_act, a_act = projectile_motion_euler(t_precise, r0, v0)


# Extract position components
r_vrlx = r_vrl[:, 0]
r_vrly = r_vrl[:, 1]
r_elrx = r_elr[:, 0]
r_elry = r_elr[:, 1]
r_rk4x = r_rk4[:, 0]
r_rk4y = r_rk4[:, 1]
r_actx = r_act[:, 0]
r_acty = r_act[:, 1]

# Plot results

plt.figure(figsize=(10, 6))
plt.plot(r_vrlx, r_vrly, color = 'r', label = 'Verlet')
plt.plot(r_elrx, r_elry, color = 'b', label = 'Euler')
plt.plot(r_rk4x, r_rk4y, color = 'g', label = 'RK4')
plt.plot(r_actx, r_acty, color = 'c', label = 'Analytical',  marker = '')

plt.ylim([-50,max(r_acty)*1.2])
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Distance (m)')
plt.title('Projectile Motion with Drag using Verlet Integration')
plt.legend()
plt.grid(True)
plt.show()
