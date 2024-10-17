import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the electric field function
def electric_field(x, y, q, pos_x, pos_y):
    """
    Calculate the electric field vector (Ex, Ey) at point (x, y) 
    due to a charge q located at (pos_x, pos_y).
    
    Parameters:
    - x, y: Coordinates of the observation point
    - q: Charge magnitude
    - pos_x, pos_y: Coordinates of the charge
    
    Returns:
    - (Ex, Ey): Electric field components at (x, y)
    """
    # Compute the displacement vector from the charge to the observation point
    r_x = x - pos_x
    r_y = y - pos_y
    r = np.sqrt(r_x**2 + r_y**2)  # Distance between charge and point
    
    # Electric field components (Coulomb's law)
    Ex = q * r_x / r**3
    Ey = q * r_y / r**3
    
    return Ex, Ey

# Step 2: Create a grid of points (x, y) over the 2D plane
x_vals = np.linspace(-5, 5, 500)
y_vals = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x_vals, y_vals)

# Step 3: Calculate the electric field at each point due to two charges
# Charge 1: Positive charge at (-1, 0)
Ex1, Ey1 = electric_field(X, Y, q=1, pos_x=-1, pos_y=0)

# Charge 2: Negative charge at (1, 0)
Ex2, Ey2 = electric_field(X, Y, q=-1, pos_x=1, pos_y=0)

# Step 4: Total electric field is the sum of the fields from both charges
Ex_total = Ex1 + Ex2
Ey_total = Ey1 + Ey2

# Step 5: Create the stream plot
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, Ex_total, Ey_total, color=np.sqrt(Ex_total**2 + Ey_total**2), cmap='inferno')

# Step 6: Add charge positions to the plot
plt.scatter([-1, 1], [0, 0], color=['red', 'blue'], s=100, label="Charges", zorder = 2)  # Red: +ve, Blue: -ve
# plt.text(-1, 0.1, '+1', color='red', fontsize=12, ha='center')
# plt.text(1, 0.1, '-1', color='blue', fontsize=12, ha='center')

# Step 7: Customize plot
plt.title("Electric Field Due to Two Charges")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid(True)
plt.colorbar(label='Electric Field Magnitude')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the electric field function
def electric_field(x, y, q, pos_x, pos_y):
    """
    Calculate the electric field vector (Ex, Ey) at point (x, y) 
    due to a charge q located at (pos_x, pos_y).
    
    Parameters:
    - x, y: Coordinates of the observation point
    - q: Charge magnitude
    - pos_x, pos_y: Coordinates of the charge
    
    Returns:
    - (Ex, Ey): Electric field components at (x, y)
    """
    r_x = x - pos_x
    r_y = y - pos_y
    r = np.sqrt(r_x**2 + r_y**2)  # Distance between charge and point
    
    # Electric field components (Coulomb's law)
    Ex = q * r_x / r**3
    Ey = q * r_y / r**3
    
    return Ex, Ey

# Step 2: Define the electric potential function
def electric_potential(x, y, q, pos_x, pos_y):
    """
    Calculate the electric potential V at point (x, y) 
    due to a charge q located at (pos_x, pos_y).
    
    Parameters:
    - x, y: Coordinates of the observation point
    - q: Charge magnitude
    - pos_x, pos_y: Coordinates of the charge
    
    Returns:
    - V: Electric potential at (x, y)
    """
    r = np.sqrt((x - pos_x)**2 + (y - pos_y)**2)  # Distance between charge and point
    return q / r

# Step 3: Create a grid of points (x, y) over the 2D plane
x_vals = np.linspace(-5, 5, 500)
y_vals = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x_vals, y_vals)

# Step 4: Calculate the electric field at each point due to two charges
# Charge 1: Positive charge at (-1, 0)
Ex1, Ey1 = electric_field(X, Y, q=1, pos_x=-1, pos_y=0)
# Charge 2: Negative charge at (1, 0)
Ex2, Ey2 = electric_field(X, Y, q=-1, pos_x=1, pos_y=0)
# Total electric field (vector sum)
Ex_total = Ex1 + Ex2
Ey_total = Ey1 + Ey2

# Step 5: Calculate the electric potential at each point due to two charges
# Potential from charge 1
V1 = electric_potential(X, Y, q=1, pos_x=-1, pos_y=0)
# Potential from charge 2
V2 = electric_potential(X, Y, q=-1, pos_x=1, pos_y=0)
# Total potential (scalar sum)
V_total = V1 + V2

# Step 6: Create the stream plot for the electric field
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, Ex_total, Ey_total, color=np.sqrt(Ex_total**2 + Ey_total**2), cmap='inferno', density=1.5)

# Step 7: Plot the equipotential lines
plt.contour(X, Y, V_total, levels=100, colors='g', linestyles='-')

# Step 8: Add charge positions to the plot
plt.scatter([-1, 1], [0, 0], color=['red', 'blue'], s=100, label="Charges", zorder=2)  # Red: +ve, Blue: -ve

# Step 9: Customize the plot
plt.title("Electric Field and Equipotential Lines for Two Charges")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid(True)
plt.colorbar(label='Electric Field Magnitude')
plt.show()
