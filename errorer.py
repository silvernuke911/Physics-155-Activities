import numpy as np 
import matplotlib.pyplot as plt 

def cost_function(y_data,y_pred):
    return np.sum((y_data - y_pred) ** 2)
    
def func(x,m,b):
    return m * x + b

m_true, b_true = 3,2 
dm = db = 0.001
vmin = 0
vmax = 5
m_list = np.arange(vmin,vmax + dm,dm)
b_list = np.arange(vmin,vmax + dm,db)

x = np.arange(0,10,0.05)
y_true = m_true * x + b_true
noise = 2 * np.random.normal(0,0.5,len(y_true))
y_data = y_true + noise

# Plot the true function for visualization
plt.plot(x, y_true, 'r', label='True Function', zorder = 2)
plt.scatter(x, y_data, color = 'k', label='Data', zorder = 3)
plt.grid()
plt.xlim(0, 10)
plt.ylim(0, 20)
plt.title("Noisy Function")
plt.show()

# Step 5: Define range for parameters 'm' and 'b'
lims = [0, vmax]  
size = 1000
m_list = np.linspace(lims[0], lims[1], size)
b_list = np.linspace(lims[0], lims[1], size)

# Step 6: Initialize error grid
error_grid = np.zeros((size, size))

# Step 7: Calculate the least squares error for each pair (a, b)
for i, m in enumerate(m_list):
    for j, b in enumerate(b_list):
        y_pred = func(x, m, b)
        # Skip if invalid (a = 0 or NaN in predictions)
        if np.isnan(y_pred).any():
            error_grid[i, j] = np.nan
        else:
            sq_error = cost_function(y_data, y_pred)
            error_grid[i, j] = sq_error

# Step 8: Visualize the error grid
plt.imshow(error_grid, cmap='hot_r', origin='lower',
           extent=[lims[0], lims[1], lims[0], lims[1]], vmax = 5000, vmin = 0)
plt.colorbar(label='Least Squares Error')
plt.xlabel('a')
plt.ylabel('b')
plt.title("Error Grid for Parameter Estimation")
plt.show()

# Find the minimum value
min_value = np.min(error_grid)
# Find the indices of the minimum value
min_index = np.unravel_index(np.argmin(error_grid), error_grid.shape)
a_guess, b_guess = m_list[min_index[0]], b_list[min_index[1]]
print("Minimum value:", min_value)
print("Indices of the minimum value:", min_index)
print(f'Minimum values at a = {a_guess} and b = {b_guess}')

y_guess = func(x,a_guess,b_guess)
# Plot the true function for visualization
plt.plot(x, y_true, 'r', label='True Function', zorder = 2)
plt.scatter(x, y_data, color = 'k', label='Data', zorder = 2)
plt.plot(x, y_guess, color = 'b', label='Minimum error function', zorder = 2)
plt.grid()
plt.xlim(0, vmax)
plt.ylim(0, 20)
plt.title("Noisy Function")
plt.show()


