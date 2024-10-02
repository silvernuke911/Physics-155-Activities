import numpy as np
import matplotlib.pyplot as plt

# Function to compute the derivative of the potential V with respect to y1
def dV_dy(y1, V):
    # Compute the derivative dV/dy using analytical methods
    h = 1e-5  # Small step for numerical differentiation
    return (V(y1 + h) - V(y1 - h)) / (2 * h)

# Function to define the system of first-order equations
def vector_field(y1, y2, V):
    # dy1/dt = y2 (since y1 is y and y2 is y')
    f1 = y2
    # dy2/dt = - dV/dy1 (the second-order equation from the potential function)
    f2 = -dV_dy(y1, V)
    return f1, f2

# Function to generate the vector field and plot it
def plot_vector_field(V):
    # Create a grid of points in the (y1, y2) plane
    dx = 0.5
    y1_vals = np.arange(-5, 5, dx)
    y2_vals = np.arange(-5, 5, dx)
    Y1, Y2 = np.meshgrid(y1_vals, y2_vals)

    # Compute the vector field at each point
    F1, F2 = vector_field(Y1, Y2, V)

    # Normalize the vectors to avoid overly large arrows
    N = np.sqrt(F1**2 + F2**2)
    F1_norm = F1 / N
    F2_norm = F2 / N

    # Plot the vector field using quiver
    plt.figure(figsize=(6, 6))
    plt.quiver(Y1, Y2, F1_norm, F2_norm, angles='xy')
    plt.xlabel('$y_1$ (y)')
    plt.ylabel('$y_2$ (y\')')
    plt.title('Vector Field for the Potential $V(y)$')
    plt.grid()
    plt.show()

def plot_potential(func):
    dx = 0.1
    x = np.arange(-5,5,dx)
    plt.plot(x,func(x))
    plt.grid()
    plt.show()

# Example 1: V(y) = 0.5 * y^2 (simple harmonic oscillator)
def potential_example_1(y1):
    return 0.5 * y1**2

# Example 2: V(y) = y^4 - y^2 (double-well potential)
def potential_example_2(y1):
    return y1**4 - y1**2

# Example 3: V(y) = y^3 (cubic potential)
def potential_example_3(y1):
    return -y1**3 + 5*y1

# Call the plot function with one of the examples
plot_potential(potential_example_2)
plot_vector_field(potential_example_2)
