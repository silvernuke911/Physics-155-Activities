import numpy as np
import matplotlib.pyplot as plt

# Exponential function model (ensuring b is positive)
def exp_model(x, a, b, c):
    return a + np.abs(b)**(x - c)  # Ensure b is positive by taking abs(b)

# Least squares cost function
def least_squares_error(params, x, y):
    a, b, c = params
    y_pred = exp_model(x, a, b, c)
    return np.sum((y - y_pred)**2)

# Gradient descent implementation
def gradient_descent(x, y, lr=1e-3, max_iter=10000, tol=1e-6):
    # Initialize parameters (a, b, c) with better guesses
    params = np.array([0.5, 2, 2])  # Initial guess close to true values
    
    # Initial error
    prev_error = least_squares_error(params, x, y)

    for _ in range(max_iter):
        # Gradient approximation (numerical gradient)
        grad = np.zeros_like(params)
        for i in range(len(params)):
            temp_params = params.copy()
            h = 1e-5
            temp_params[i] += h
            grad[i] = (least_squares_error(temp_params, x, y) - prev_error) / h
        
        # Update parameters using gradient descent
        params -= lr * grad
        
        # Calculate the new error
        curr_error = least_squares_error(params, x, y)
        
        # Check for convergence
        if abs(prev_error - curr_error) < tol:
            break
        
        prev_error = curr_error
    
    return params

# Generate synthetic data for testing
np.random.seed(42)
x = np.linspace(0, 5, 100)
a_true, b_true, c_true = 0.5, 2, 2  # True parameters
y_true = exp_model(x, a_true, b_true, c_true)
y_data = y_true + 0.1 * np.random.randn(len(x))  # Add noise to the data

# Fit the data using gradient descent
best_params = gradient_descent(x, y_data)

# Print the best fit parameters
print(f"Best fit parameters: a = {best_params[0]}, b = {np.abs(best_params[1])}, c = {best_params[2]}")

# Plot the original data and the fitted curve
plt.scatter(x, y_data, label='Noisy Data', color='red')
plt.plot(x, y_true, label='True Curve', color='g')
plt.plot(x, exp_model(x, *best_params), label='Fitted Curve', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential Fit Using Gradient Descent (Improved)')
plt.ylim(0,10)
plt.legend()
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Exponential function model
def exp_model(x, a, b, c):
    return a + b * np.exp(x - c)

# Generate synthetic data for testing
np.random.seed(42)
x = np.linspace(0, 5, 100)
a_true, b_true, c_true = 0.5, 2, 2  # True parameters
y_true = exp_model(x, a_true, b_true, c_true)
y_data = y_true + 0.5 * np.random.randn(len(x))  # Add noise to the data

# Fit the data using scipy curve_fit
initial_guess = [1, 1, 1]  # Initial guess for a, b, c
popt, pcov = curve_fit(exp_model, x, y_data, p0=initial_guess)

# Extract the best fit parameters
a_fit, b_fit, c_fit = popt
print(f"Best fit parameters: a = {a_fit}, b = {b_fit}, c = {c_fit}")

# Plot the original data and the fitted curve
plt.scatter(x, y_data, label='Noisy Data', color='red')
plt.plot(x, exp_model(x, *popt), label='Fitted Curve', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential Fit Using SciPy')
plt.legend()
plt.ylim(0,10)
plt.grid()
plt.show()
