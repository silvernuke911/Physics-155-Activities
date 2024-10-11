import numpy as np 
import matplotlib.pyplot as plt 

def forward_euler(f, x, y_0, dx):
    y = np.zeros_like(x) 
    y[0] = y_0
    for i in range(len(x) - 1):
        y[i+1] = y[i] + f(x[i], y[i]) * dx  
    return y

def mid_point_method(f, x, y_0, dx):
    y = np.zeros_like(x) 
    y[0] = y_0
    for i in range(len(x) - 1):
        y[i + 1] = y[i] + dx * f(x[i] + dx/2, y[i] + (dx/2) * f(x[i], y[i]))
    return y

def rk4(f, x, y_0, dx):
    y = np.zeros_like(x) 
    y[0] = y_0
    for i in range(len(x) - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + dx/2, y[i] + dx*k1/2)
        k3 = f(x[i] + dx/2, y[i] + dx*k2/2)
        k4 = f(x[i] + dx, y[i] + dx*k3)
        y[i + 1] = y[i] + (dx/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y

def rk2(f, x, y_0, dx):
    y = np.zeros_like(x) 
    y[0] = y_0
    for i in range(len(x) - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + dx/2, y[i] + dx*k1/2)
        y[i + 1] = y[i] + dx * k2
    return y

def func(x, y):
    return y

def real_sol(x):
    return np.exp(x)

# Initial conditions and setup
x0, xf, dx = 0.1, 100, 0.05 # Adjusted x0 for better log scaling
y0 = real_sol(x0)
x = np.arange(x0, xf + dx, dx)  # Ensure the last point is included

# Compute solutions
real = real_sol(x)
fwde = forward_euler(func, x, y0, dx)
mdpt = mid_point_method(func, x, y0, dx)
# rk2_ = rk2(func, x, y0, dx)
rk4_ = rk4(func, x, y0, dx)

# Plot solutions

plt.plot(x, real, color='r', label='Exact Solution')
plt.plot(x, fwde, color='b', label='Forward Euler')
plt.plot(x, mdpt, color='g', label='Midpoint Method')
# plt.plot(x, rk2_, color='m', label='RK2')
plt.plot(x, rk4_, color='c', label='RK4')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Numerical Methods')
plt.legend()
plt.grid()
plt.show()

# Compute errors
fwde_error = np.abs(fwde - real)
mdpt_error = np.abs(mdpt - real)
# rk2__error = np.abs(rk2_ - real)
rk4__error = np.abs(rk4_ - real)

# Avoid log scaling with zero values
def safe_log_scale(y):
    return np.where(y > 0, y, np.nan)

# Plot errors
plt.figure(figsize=(10, 5))
plt.plot(x, safe_log_scale(fwde_error), color='b', label='Forward Euler Error')
plt.plot(x, safe_log_scale(mdpt_error), color='g', label='Midpoint Method Error')
# plt.plot(x, safe_log_scale(rk2__error), color='m', label='RK2 Error')
plt.plot(x, safe_log_scale(rk4__error), color='c', label='RK4 Error')

plt.xscale('log')  # Ensure x values are suitable for log scale
plt.yscale('log')  # Apply log scale to y-axis
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Error Comparison')
plt.legend()
plt.grid()
plt.show()

# Compute relative errors
fwde_error = np.abs((fwde - real)/real)
mdpt_error = np.abs((mdpt - real)/real)
# rk2__error = np.abs((rk2_ - real)/real)
rk4__error = np.abs((rk4_ - real)/real)

# Avoid log scaling with zero values
def safe_log_scale(y):
    return np.where(y > 0, y, np.nan)

# Plot errors
plt.figure(figsize=(10, 5))
plt.plot(x, safe_log_scale(fwde_error), color='b', label='Forward Euler Error')
plt.plot(x, safe_log_scale(mdpt_error), color='g', label='Midpoint Method Error')
# plt.plot(x, safe_log_scale(rk2__error), color='m', label='RK2 Error')
plt.plot(x, safe_log_scale(rk4__error), color='c', label='RK4 Error')

plt.xscale('log')  # Ensure x values are suitable for log scale
plt.yscale('log')  # Apply log scale to y-axis
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Error Comparison')
plt.legend()
plt.grid()
plt.show()
