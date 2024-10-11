import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return x**2 / 2

# Parameters
guess = 3
max_iter = 10
tol = 1e-6

x_coord = np.zeros(max_iter)
y_coord = np.zeros(max_iter)
def fixed_point_iteration(func, x0, tol=1e-7, max_iter=1000):
    x_n = x0
    x_coord[0] = x_n
    y_coord[0] = 0
    for n in range(1, max_iter):
        if x_n > func(x_n):
            x_n1 = func(x_n)
        else:
            x_n1 = x_n
        if abs(x_n1 - x_n) < tol:  # Check for convergence
            return x_n1, n
        x_n = x_n1  # Update x for next iteration
    raise ValueError("Fixed Point Iteration did not converge.")
x = fixed_point_iteration(func,guess)
print(f'Fixed point: {x}')

print(x_coord)
print(y_coord)
x_pos = np.arange(0,5,0.001)
plt.plot(x_pos,func(x_pos))
plt.plot(x_pos,x_pos)
plt.grid()
plt.xlim([0,5])
plt.ylim([0,5])
plt.gca().set_aspect('equal')
plt.show()
