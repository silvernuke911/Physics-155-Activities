import numpy as np

def newtons_method(func, x0, tol=1e-10, max_iter=100, h=1e-10):
    """
    Newton's method for finding the root of a function using a numerical derivative if not provided.
    
    Parameters:
    - func: The function for which the root is to be found.
    - x0: Initial guess for the root.
    - tol: Tolerance for stopping the iteration.
    - max_iter: Maximum number of iterations.
    - h: Small value for numerical derivative approximation.
    
    Returns:
    - The estimated root of the function.
    """
    def numerical_derivative(f, x, h):
        return (f(x + h) - f(x - h)) / (2*h)
    x = x0
    for _ in range(max_iter):
        f_val = func(x)
        df_val = numerical_derivative(func, x, h)
        if df_val == 0:
            print("Derivative is zero. Stopping iteration to avoid division by zero.")
            return None
        x_new = x - f_val / df_val
        # Check if the difference is within tolerance
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    print("Exceeded maximum iterations.")
    return x
# Example usage with a sample function
def func(x):
    return np.cos(x)  # Example: f(x) = x^2 - 2, root is sqrt(2)

# Initial guess
x0 = 1.0
root = newtons_method(func, x0)
print(f"Estimated root: {root}")
print(np.pi/2)