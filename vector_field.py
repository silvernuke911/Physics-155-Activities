import numpy as np
import matplotlib.pyplot as plt

# Define the system of equations
def vector_field(y1, y2):
    f1 = y2
    f2 = -y1
    return f1, f2

# Create a grid of points in the (y1, y2) plane
y1_vals = np.linspace(-5, 5, 20)
y2_vals = np.linspace(-5, 5, 20)
Y1, Y2 = np.meshgrid(y1_vals, y2_vals)

# Compute the vector field at each point
F1, F2 = vector_field(Y1, Y2)

# Normalize the vectors to avoid overly large arrows
N = np.sqrt(F1**2 + F2**2)
F1_norm = F1 / N
F2_norm = F2 / N

# Plot the vector field using quiver
plt.figure(figsize=(6, 6))
plt.quiver(Y1, Y2, F1_norm, F2_norm, angles='xy')
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('Vector Field of the System')
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Function to define the system of first-order equations
def vector_field(y1, y2, func):
    # dy1/dt = y2 (since y1 is y and y2 is y')
    f1 = y2
    # dy2/dt = func(y1, y2) (this is the second-order function you provide)
    f2 = func(y1, y2)
    return f1, f2

# Function to generate the vector field and plot it
def plot_vector_field(func):
    # Create a grid of points in the (y1, y2) plane
    dx = 0.5
    y1_vals = np.arange(-5, 5, dx)
    y2_vals = np.arange(-5, 5, dx)
    Y1, Y2 = np.meshgrid(y1_vals, y2_vals)

    # Compute the vector field at each point
    F1, F2 = vector_field(Y1, Y2, func)

    # Normalize the vectors to avoid overly large arrows
    N = np.sqrt(F1**2 + F2**2)
    F1_norm = F1 / N
    F2_norm = F2 / N

    # Plot the vector field using quiver
    plt.figure(figsize=(6, 6))
    plt.quiver(Y1, Y2, F1_norm, F2_norm, angles='xy')
    plt.xlabel('$y_1$ (y)')
    plt.ylabel('$y_2$ (y\')')
    plt.title('Vector Field for $y\'\' = f(y, y\')$')
    plt.grid()
    plt.show()

# Example 1: y'' = -y (simple harmonic oscillator)
def func_example_1(y1, y2):
    return -y1

# Example 2: y'' = y' (exponential growth/decay)
def func_example_2(y1, y2):
    return y2

# Example 3: y'' = y' - y (damped harmonic oscillator)
def func_example_3(y1, y2):
    return -0.5*y2 - y1

# Call the plot function with one of the examples
plot_vector_field(func_example_3)
