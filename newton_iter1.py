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

def poly_func(x, coef):
    y = np.zeros_like(x, dtype=float)
    degree = len(coef) - 1
    for n, c in enumerate(coef):
        y += c * x**(degree - n)
    return y

# New function: Newton's method for finding extrema (maxima and minima)
def newton_extrema(func, guess_range, tol=1e-16, max_iter=100, h=1e-10):
    """
    Find the extrema (maxima and minima) of a function using Newton's method.
    
    Parameters:
    - func: The original function whose extrema are to be found.
    - guess_range: The range of initial guesses for finding the extrema.
    - tol: Tolerance for stopping Newton's method.
    - max_iter: Maximum number of iterations for Newton's method.
    - h: Step size for numerical derivative approximation.
    
    Returns:
    - extrema_points: List of critical points where extrema occur.
    """
    def derivative(f, x, h):
        return (f(x + h) - f(x - h)) / (2 * h)

    def second_derivative(f, x, h):
        return (f(x + h) - 2 * f(x) + f(x - h)) / h**2

    extrema_points = []

    # Apply Newton's method to the derivative of the function
    for guess in guess_range:
        root = newtons_method(lambda x: derivative(func, x, h), guess, tol=tol, max_iter=max_iter, h=h)
        if root is not None:
            if not np.isnan(root) and root not in extrema_points:
                # Check if it's a maximum or minimum using the second derivative test
                if second_derivative(func, root, h) > 0:
                    print(f"Minima found at x = {root}")
                elif second_derivative(func, root, h) < 0:
                    print(f"Maxima found at x = {root}")
                else:
                    print(f"Saddle point found at x = {root}")
                extrema_points.append(root)

    # Remove duplicates and return unique extrema points
    return np.array(list(set(np.round(extrema_points, decimals=8))))

# Example usage with a sample function
# Define the original polynomial function
coefficients = [0.1, 1.2, 3.2, 1]
actual_roots = np.roots(coefficients)

# Example function to analyze
def func(x):
    return poly_func(x, coefficients)

# Initial guesses for extrema from -10 to 10
guess_range = np.arange(-10, 10, 0.5)
extrema_points = newton_extrema(func, guess_range)

print(f"Estimated extrema points : {extrema_points}")

# Plotting
x = np.arange(-10, 10, 0.001)
plt.grid()
plt.plot(x, func(x), 'r', zorder=2, label='Function')
plt.scatter(extrema_points, func(extrema_points), color='g', zorder=3, label="Extrema Points")
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.gca().set_aspect('equal')
plt.legend()
plt.show()
