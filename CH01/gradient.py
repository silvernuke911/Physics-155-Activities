import numpy as np

def manual_gradient(f, x, y):
    # Get the shape of the grid
    m, n = f.shape
    
    # Initialize arrays to store the partial derivatives (gradients)
    df_dx = np.zeros_like(f)
    df_dy = np.zeros_like(f)
    
    # Compute the step size in x and y directions
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Compute partial derivatives using finite differences
    for i in range(m):
        for j in range(n):
            # Compute df/dx (partial derivative in x direction)
            if i == 0:
                df_dx[i, j] = (f[i+1, j] - f[i, j]) / dx  # forward difference
            elif i == m - 1:
                df_dx[i, j] = (f[i, j] - f[i-1, j]) / dx  # backward difference
            else:
                df_dx[i, j] = (f[i+1, j] - f[i-1, j]) / (2 * dx)  # central difference

            # Compute df/dy (partial derivative in y direction)
            if j == 0:
                df_dy[i, j] = (f[i, j+1] - f[i, j]) / dy  # forward difference
            elif j == n - 1:
                df_dy[i, j] = (f[i, j] - f[i, j-1]) / dy  # backward difference
            else:
                df_dy[i, j] = (f[i, j+1] - f[i, j-1]) / (2 * dy)  # central difference

    return df_dx, df_dy

# Example usage
# Define a 2D function f(x, y)
x = np.linspace(0, 2 * np.pi, 100)
y = np.linspace(0, 2 * np.pi, 100)
X, Y = np.meshgrid(x, y)
f = np.sin(X) * np.cos(Y)

# Compute the gradient manually
df_dx, df_dy = manual_gradient(f, x, y)

# Display the results
print("df/dx:\n", df_dx)
print("df/dy:\n", df_dy)