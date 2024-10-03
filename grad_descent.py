import numpy as np

def gradient_descent(func, x0, y0, learning_rate=0.01, tol=1e-10, max_iter=1000, h=1e-10):
    """
    Gradient descent for minimizing a 2D function using numerical derivatives.
    
    Parameters:
    - func: The 2D function to minimize, takes (x, y) as inputs.
    - x0: Initial guess for x.
    - y0: Initial guess for y.
    - learning_rate: Step size for each iteration.
    - tol: Tolerance for stopping criteria.
    - max_iter: Maximum number of iterations.
    - h: Small value for numerical derivative approximation.
    
    Returns:
    - x, y: The values of x and y that minimize the function.
    """
    def numerical_derivative_x(f, x, y, h):
        return (f(x + h, y) - f(x - h, y)) / (2 * h)

    def numerical_derivative_y(f, x, y, h):
        return (f(x, y + h) - f(x, y - h)) / (2 * h)
    
    x, y = x0, y0

    for i in range(max_iter):
        # Calculate numerical derivatives
        grad_x = numerical_derivative_x(func, x, y, h)
        grad_y = numerical_derivative_y(func, x, y, h)

        # Update x and y using the learning rate and gradient
        x_new = x - learning_rate * grad_x
        y_new = y - learning_rate * grad_y

        # Check if the updates are smaller than the tolerance
        if np.sqrt((x_new - x) ** 2 + (y_new - y) ** 2) < tol:
            return x_new, y_new
        
        x, y = x_new, y_new
    
    print("Exceeded maximum iterations.")
    return x, y

# Example 2D function: f(x, y) = (x-2)^2 + (y-3)^2 (minimum at x=2, y=3)
def func(x, y):
    return (x - 2)**2 + (y - 3)**2

# Initial guess for (x, y)
x0, y0 = 0.0, 0.0

# Run gradient descent
x_min, y_min = gradient_descent(func, x0, y0)
print(f"Minimum found at x = {x_min}, y = {y_min}")