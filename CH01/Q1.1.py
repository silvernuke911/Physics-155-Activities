# 1.1
def fact(n):
    p = 1
    if n < 0 or isinstance(n, float):
        raise ValueError('Negative or floating point numbers undefined')
    if n == 0:
        return 1
    for i in range(1,n+1):
        p*=i
    return p
print(fact(1))

def nonfloatsum(numlist):
    s,e = 0.,0.
    for x in numlist:
        temp = s 
        y = x + e 
        s = temp + y 
        e = (temp - s) + y
    return s

print(nonfloatsum([0.1,0.2]))

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
print(coefficients)
# Step 3: Solve the characteristic polynomial for eigenvalues
eigenvalues = np.roots(coefficients)

# Step 4: Display the eigenvalues
print("Eigenvalues:", eigenvalues)