import numpy as np
import matplotlib.pyplot as plt

def kinematic_rk4(t, f, r0, v0):
    dt = t[1] - t[0]  # Assuming uniform time steps
    half_dt = dt / 2
    # Initialize arrays to store positions, velocities, and accelerations
    dim = r0.ndim
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

def spring_acc(t,r,v):
    k = 0.5
    m = 1
    b = 2
    return - (k/m) * r - b * v

t0,tf,dt = 0,100,0.01

t = np.arange(t0,tf,dt)
r0 = np.array([1])
v0 = np.array([0])

simulation = kinematic_rk4(t,spring_acc,r0,v0)
y = simulation[0]
v = simulation[1]
a = simulation[2]

plt.plot(t,y, color = 'r')
plt.plot(t,v, color = 'b')
plt.plot(t,a, color = 'g')
plt.grid()
plt.show()

def x_func(t,k,b,m, amp, delta):
    omega = np.sqrt(k/m)
    beta = b / (2*m)
    return amp * np.exp(-beta * t) * np.cos(omega * t - delta)
y_real = x_func(t,0.5,0.1,1,1,0)
plt.plot(t,y, color = 'r')
plt.plot(t,y_real, color = 'b')
plt.grid()
plt.show()

plt.plot(y,v,color='black')
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot{x}$')
plt.grid()
plt.show()
