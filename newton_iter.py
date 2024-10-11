import numpy as np
import matplotlib.pyplot as plt

def newtons_method(func, x0, tol=1e-16, max_iter=100, h=1e-10):
    """
    Newton's method for finding the root of a function using a numerical derivative.
    
    Parameters:
    - func: The function for which the root is to be found.
    - x0: Initial guess for the root.
    - tol: Tolerance for stopping the iteration.
    - max_iter: Maximum number of iterations.
    - h: Small value for numerical derivative approximation.
    
    Returns:
    - The estimated root of the function, or None if it does not converge.
    """
    def numerical_derivative(f, x, h):
        return (f(x + h) - f(x - h)) / (2 * h)
    
    x = x0
    for _ in range(max_iter):
        f_val = func(x)
        df_val = numerical_derivative(func, x, h)
        if df_val == 0:
            print(f"Derivative is zero at x = {x}. Stopping iteration to avoid division by zero.")
            return None
        x_new = x - f_val / df_val
        # Check if the difference is within tolerance
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    print(f"Exceeded maximum iterations at x = {x}")
    return x

# Example usage with a sample function
# Calculate actual roots for comparison using np.roots
coefficients = [0.1, 1.2, 3.2, 1] 
actual_roots = np.roots(coefficients)

def func(x):
    return x**2 + 2*x - 2  # Example: f(x) = x^2 + 2x - 2

def poly_func(x, coef):
    y = np.zeros_like(x,  dtype=float)  
    degree = len(coef) - 1  
    for n, c in enumerate(coef):
        y += c * x**(degree - n) 
    return y

# Initial guesses from -10 to 9
guesses = np.arange(-10, 10)
roots = np.zeros_like(guesses, dtype=float)

# Use lambda to pass the polynomial function with fixed coefficients to Newton's method
for i, guess in enumerate(guesses):
    root = newtons_method(lambda x: poly_func(np.array([x]), coef=coefficients)[0], guess)
    if root is not None:
        roots[i] = root
    else:
        roots[i] = np.nan  # Use NaN for non-converging cases

# Use set to remove duplicate roots, ignoring NaNs
unique_roots = np.array(list(set(np.round(roots[~np.isnan(roots)], decimals=8)))) # Filter out NaNs and round for comparison

print(f"Estimated unique roots : {unique_roots}")
print(f"Actual roots           : {actual_roots}")

# Plotting
x = np.arange(-10, 10, 0.001)
plt.grid()
plt.plot(x, poly_func(x, coefficients), 'r', zorder=2)
plt.scatter(unique_roots, np.zeros_like(unique_roots), color='b', zorder=3, label="Actual Roots")
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.gca().set_aspect('equal')
plt.legend()
plt.show()
