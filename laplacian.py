import numpy as np

def manual_laplacian(f, x, y):
    # Get the shape of the grid
    m, n = f.shape
    
    # Initialize array to store the Laplacian
    laplacian = np.zeros_like(f)
    
    # Compute the step sizes in x and y directions
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Compute the Laplacian using finite differences
    for i in range(m):
        for j in range(n):
            # Compute second derivative in x-direction (d^2f/dx^2)
            if i == 0:
                d2f_dx2 = (f[i+1, j] - 2*f[i, j]) / dx**2  # forward difference
            elif i == m - 1:
                d2f_dx2 = (f[i-1, j] - 2*f[i, j]) / dx**2  # backward difference
            else:
                d2f_dx2 = (f[i+1, j] - 2*f[i, j] + f[i-1, j]) / dx**2  # central difference

            # Compute second derivative in y-direction (d^2f/dy^2)
            if j == 0:
                d2f_dy2 = (f[i, j+1] - 2*f[i, j]) / dy**2  # forward difference
            elif j == n - 1:
                d2f_dy2 = (f[i, j-1] - 2*f[i, j]) / dy**2  # backward difference
            else:
                d2f_dy2 = (f[i, j+1] - 2*f[i, j] + f[i, j-1]) / dy**2  # central difference

            # The Laplacian is the sum of the second derivatives in x and y directions
            laplacian[i, j] = d2f_dx2 + d2f_dy2

    return laplacian

# Example usage
# Define a 2D function f(x, y)
x = np.linspace(0, 2 * np.pi, 100)
y = np.linspace(0, 2 * np.pi, 100)
X, Y = np.meshgrid(x, y)
f = np.sin(X) * np.cos(Y)

# Compute the Laplacian manually
laplacian = manual_laplacian(f, x, y)

# Display the result
print("Laplacian:\n", laplacian)