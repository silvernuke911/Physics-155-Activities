import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the function to compute the Mandelbrot set
def mandelbrot(c, max_iter):
    """
    Determine whether a point 'c' is in the Mandelbrot set.
    
    Parameters:
    - c: Complex number representing a point on the complex plane.
    - max_iter: Maximum number of iterations.
    
    Returns:
    - The number of iterations before divergence (bounded by max_iter).
    """
    z = 0
    for n in range(max_iter):
        z = z**2 + c
        if abs(z) > 2:
            return n
    return max_iter

# Step 2: Set up the grid of points (the complex plane)
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Generate a 2D array representing the Mandelbrot set.
    
    Parameters:
    - xmin, xmax, ymin, ymax: The bounds of the region in the complex plane.
    - width, height: The resolution of the grid.
    - max_iter: Maximum number of iterations.
    
    Returns:
    - A 2D array where each value represents the number of iterations.
    """
    # Create a grid of complex numbers
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    mandelbrot_grid = np.zeros((height, width))
    
    for i in range(width):
        for j in range(height):
            c = real[i] + 1j * imag[j]  # Complex number c = x + iy
            mandelbrot_grid[j, i] = mandelbrot(c, max_iter)
    
    return mandelbrot_grid

# Step 3: Define parameters and generate the Mandelbrot set
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
resolution = 2000
width, height = resolution, resolution  # Resolution of the grid
max_iter = 200  # Maximum iterations

# Generate the Mandelbrot set
mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)

# Step 4: Plot the Mandelbrot set
plt.figure(figsize=(10, 10))
plt.imshow(mandelbrot_image, extent=[xmin, xmax, ymin, ymax], cmap='hot')
plt.colorbar(label='Iterations before divergence')
plt.title('Mandelbrot Set')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.show()
