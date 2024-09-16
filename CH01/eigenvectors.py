import numpy as np

# Gaussian elimination to row-reduce the matrix to RREF form
def gaussian_elimination(mat):
    n = len(mat)
    for i in range(n):
        # Make the diagonal element 1
        mat[i] = mat[i] / mat[i][i]
        
        # Make the elements below the pivot in the same column 0
        for j in range(i + 1, n):
            mat[j] = mat[j] - mat[j][i] * mat[i]
    
    # Back substitution to make the matrix upper triangular
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            mat[j] = mat[j] - mat[j][i] * mat[i]

    return mat

# Function to find eigenvectors without np.linalg
def find_eigenvectors_manual(A, eigenvalues):
    n = A.shape[0]
    I = np.eye(n)
    eigenvectors = []

    for lambda_val in eigenvalues:
        # Form (A - λI)
        matrix = A - lambda_val * I
        
        # Perform Gaussian elimination to reduce (A - λI)
        reduced_matrix = gaussian_elimination(matrix.copy())

        # The null space of the reduced matrix gives us the eigenvector
        # We'll assume it's the row with all zeros and find a free variable
        eigenvector = np.zeros(n)
        for i in range(n):
            if np.all(reduced_matrix[i] == 0):  # Check if a row is all zeros
                eigenvector[i] = 1  # Assign a free variable to the eigenvector

        eigenvectors.append(eigenvector)
    
    return np.array(eigenvectors)

# Example matrix
A = np.array([[4, -2],
              [1,  1]])

# Let's assume we already computed the eigenvalues (e.g., [3, 2])
eigenvalues = [3, 2]

# Find the eigenvectors manually
eigenvectors = find_eigenvectors_manual(A, eigenvalues)
print("Eigenvectors:", eigenvectors)