import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the function
def func(x, a, b):
    # Ensure 'a' is positive and non-zero
    if a <= 0:
        return np.nan  # Return NaN if 'a' is invalid (though handled outside)
    return a ** (x - b)

# Step 2: Define the least squares error function
def least_squares_error(y_data, y_pred):
    return np.sum((y_data - y_pred) ** 2)

# Step 3: Set true values for parameters
a_true, b_true = 2, 1
dx = 0.1
xmax = 5
x = np.arange(0, xmax + dx, dx)

# Step 4: Generate the true data using the true parameters
y_true = func(x, a_true, b_true)
noise = 2 * np.random.uniform(-1,1,len(y_true))
y_data = y_true + noise

# Plot the true function for visualization
plt.plot(x, y_true, 'r', label='True Function')
plt.scatter(x, y_data, color = 'k', label='Data')
plt.grid()
plt.xlim(0, xmax)
plt.ylim(0, 20)
plt.title("Noisy Function")
plt.show()

# Step 5: Define range for parameters 'a' and 'b'
lims = [0.01, xmax]  # Avoid 0 for 'a' since it causes issues
size = 1000
a_list = np.linspace(lims[0], lims[1], size)
b_list = np.linspace(lims[0], lims[1], size)

# Step 6: Initialize error grid
error_grid = np.zeros((size, size))

# Step 7: Calculate the least squares error for each pair (a, b)
for i, a in enumerate(a_list):
    for j, b in enumerate(b_list):
        y_pred = func(x, a, b)
        # Skip if invalid (a = 0 or NaN in predictions)
        if np.isnan(y_pred).any():
            error_grid[i, j] = np.nan
        else:
            sq_error = least_squares_error(y_data, y_pred)
            error_grid[i, j] = sq_error

# Step 8: Visualize the error grid
plt.imshow(error_grid, cmap='hot_r', origin='lower',
           extent=[lims[0], lims[1], lims[0], lims[1]], vmax = 500, vmin = 0)
plt.colorbar(label='Least Squares Error')
plt.xlabel('a')
plt.ylabel('b')
plt.title("Error Grid for Parameter Estimation")
plt.show()

# Find the minimum value
min_value = np.min(error_grid)
# Find the indices of the minimum value
min_index = np.unravel_index(np.argmin(error_grid), error_grid.shape)
a_guess, b_guess = a_list[min_index[0]], b_list[min_index[1]]
print("Minimum value:", min_value)
print("Indices of the minimum value:", min_index)
print(f'Minimum values at a = {a_guess} and b = {b_guess}')

y_guess = func(x,a_guess,b_guess)
# Plot the true function for visualization
plt.plot(x, y_true, 'r', label='True Function', zorder = 2)
plt.scatter(x, y_data, color = 'k', label='Data', zorder = 2)
plt.plot(x, y_guess, color = 'b', label='Minimum error function', zorder = 2)
plt.grid()
plt.xlim(0, xmax)
plt.ylim(0, 20)
plt.title("Noisy Function")
plt.show()
