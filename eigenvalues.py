import numpy as np

def manual_poly(A):
    # Get the size of the matrix
    n = A.shape[0]
    
    # Start with the identity matrix (symbolically A - λI)
    I = np.eye(n)

    # Compute the coefficients of the characteristic polynomial
    # λ^n - trace(A)*λ^(n-1) + ... + (-1)^n * det(A)
    coefficients = []
    
    # We need the determinant of (A - λI) for different λ
    for i in range(n + 1):
        # Compute (A - λI) by scaling I with λ and subtracting from A
        matrix = A - (i * I)
        
        # Compute the determinant of the matrix
        det_val = np.linalg.det(matrix)
        
        # Add the determinant value to the coefficients (the signs alternate)
        coefficients.append((-1)**i * det_val)
    
    return np.array(coefficients)

# Example matrix
A = np.array([[4, -2],
              [1,  1]])

coefficients = manual_poly(A)
print("Characteristic Polynomial Coefficients:", coefficients)

import numpy as np

# Step 1: Define the matrix
A = np.array([[4, -2],
              [1,  1]])

# Step 2: Compute the characteristic polynomial
# Create an identity matrix of the same size as A
I = np.eye(A.shape[0])

# Subtract λ*I from A (symbolic lambda here)
# In practice, you need to find the determinant of (A - λI)
# Using the fact that det(A - λI) gives the characteristic polynomial
coefficients = np.poly(A)  # This gives the coefficients of the characteristic polynomial

# Step 3: Solve the characteristic polynomial for eigenvalues
eigenvalues = np.roots(coefficients)

# Step 4: Display the eigenvalues
print("Eigenvalues:", eigenvalues)

import numpy as np

def manual_roots(coefficients):
    # Ensure the leading coefficient is not zero
    if coefficients[0] == 0:
        raise ValueError("The leading coefficient cannot be zero.")
    
    # Degree of the polynomial
    degree = len(coefficients) - 1
    
    # Normalize the coefficients by dividing by the leading coefficient
    coefficients = coefficients / coefficients[0]
    
    # Construct the companion matrix
    companion_matrix = np.zeros((degree, degree))
    
    # Fill the first sub-diagonal with 1's
    for i in range(1, degree):
        companion_matrix[i-1, i] = 1
    
    # Fill the last row with the negative coefficients (a_0/a_n, a_1/a_n, ..., a_{n-1}/a_n)
    companion_matrix[-1, :] = -coefficients[1:]
    
    # Compute the eigenvalues of the companion matrix (these are the roots)
    eigenvalues = np.linalg.eigvals(companion_matrix)
    
    return eigenvalues

# Example usage
coefficients = [1, -6, 11, -6]  # This represents the polynomial x^3 - 6x^2 + 11x - 6
roots = manual_roots(coefficients)
print("Roots:", roots)