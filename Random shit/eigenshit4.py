import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 10  # Domain length
n = 100  # Number of grid points
dx = L / (n - 1)
x = np.linspace(0, L, n)

# Create the double derivative operator (second-order finite difference)
def double_derivative_operator(n, dx):
    main_diag = -2 * np.ones(n)
    side_diag = np.ones(n - 1)
    
    # Create the matrix
    A = np.diag(main_diag) + np.diag(side_diag, -1) + np.diag(side_diag, 1)
    
    # Apply Dirichlet boundary conditions: y(0) = y(L) = 0
    A[0, :] = 0
    A[-1, :] = 0
    A[0, 0] = 1  # Set boundary conditions explicitly
    A[-1, -1] = 1
    
    return A / dx**2

# Construct the second derivative operator
A = double_derivative_operator(n, dx)

# Solve the eigenvalue problem A @ y = λ * y
eigenvalues, eigenvectors = np.linalg.eig(A)

# Sort eigenvalues and corresponding eigenvectors
idx = eigenvalues.argsort()  # Sort eigenvalues and eigenvectors
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Choose the first few eigenvalues (since we sort them)
print("First few eigenvalues:", eigenvalues[:5])

# Plot the first few eigenfunctions
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(x, eigenvectors[:, i], label=f"Eigenfunction {i+1} (λ = {eigenvalues[i]:.2f})")
    
plt.title('Eigenfunctions of Second Derivative Operator')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 10  # Domain length
n = 1000  # Number of grid points
dx = L / (n - 1)
x = np.linspace(0, L, n)

# Create the double derivative operator (second-order finite difference)
def double_derivative_operator(n, dx, bc_type="dirichlet"):
    main_diag = -2 * np.ones(n)
    side_diag = np.ones(n - 1)
    
    # Create the matrix
    A = np.diag(main_diag) + np.diag(side_diag, -1) + np.diag(side_diag, 1)
    
    # Apply boundary conditions
    if bc_type == "dirichlet":
        A[0, :] = 0  # Dirichlet condition at left boundary
        A[0, 0] = 1
        A[-1, :] = 0  # Dirichlet condition at right boundary
        A[-1, -1] = 1
    elif bc_type == "neumann":
        A[0, 0] = -2
        A[0, 1] = 2  # Forward difference at left boundary
        A[-1, -2] = -2
        A[-1, -1] = 2  # Backward difference at right boundary
    
    return A / dx**2

# Construct the second derivative operator with Dirichlet boundary conditions
A = double_derivative_operator(n, dx, bc_type="dirichlet")

# Solve the eigenvalue problem A @ y = λ * y
eigenvalues, eigenvectors = np.linalg.eig(A)

# Sort eigenvalues and corresponding eigenvectors
idx = eigenvalues.argsort()  # Sort eigenvalues and eigenvectors
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Specify the target eigenvalue (or choose the closest one)
target_lambda = -0.99  # Example: Close to -1
closest_index = (np.abs(eigenvalues - target_lambda)).argmin()

# Extract the corresponding eigenvalue and eigenfunction
closest_eigenvalue = eigenvalues[closest_index]
closest_eigenfunction = eigenvectors[:, closest_index]

# Print the selected eigenvalue
print(f"Selected Eigenvalue: {closest_eigenvalue}")

# Plot the eigenfunction corresponding to the chosen eigenvalue
plt.figure(figsize=(8, 6))
plt.plot(x, closest_eigenfunction, label=f"Eigenfunction for λ = {closest_eigenvalue:.3f}")
plt.title(f'Eigenfunction Corresponding to Eigenvalue λ = {closest_eigenvalue:.3f}')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True)
plt.legend()
plt.show()
