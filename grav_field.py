import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the gravitational field function
def gravitational_field(x, y, m, pos_x, pos_y):
    """
    Calculate the gravitational field vector (gx, gy) at point (x, y) 
    due to a mass m located at (pos_x, pos_y).
    
    Parameters:
    - x, y: Coordinates of the observation point
    - m: Mass
    - pos_x, pos_y: Coordinates of the mass
    
    Returns:
    - (gx, gy): Gravitational field components at (x, y)
    """
    # Compute the displacement vector from the mass to the observation point
    r_x = x - pos_x
    r_y = y - pos_y
    r = np.sqrt(r_x**2 + r_y**2)  # Distance between mass and point
    
    # Gravitational field components (inversely proportional to r^2)
    # We use -m to represent the attractive nature of gravity
    gx = -m * r_x / r**3
    gy = -m * r_y / r**3
    
    return gx, gy

# Step 2: Create a grid of points (x, y) over the 2D plane
x_vals = np.linspace(-3, 3, 500)
y_vals = np.linspace(-3, 3, 500)
X, Y = np.meshgrid(x_vals, y_vals)

# Step 3: Calculate the gravitational field at each point due to two masses
# Mass 1: m1 located at (-1, 0)
gx1, gy1 = gravitational_field(X, Y, m=10, pos_x=-1, pos_y=0)

# Mass 2: m2 located at (1, 0)
gx2, gy2 = gravitational_field(X, Y, m=1, pos_x=1, pos_y=0)

# Step 4: Total gravitational field is the sum of the fields from both masses
gx_total = gx1 + gx2
gy_total = gy1 + gy2

# Step 5: Create the stream plot
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, gx_total, gy_total, color=np.sqrt(gx_total**2 + gy_total**2), cmap='viridis')

# Step 6: Add mass positions to the plot
plt.scatter([-1, 1], [0, 0], color=['orange', 'green'], s=100, label="Masses", zorder = 2)  # Orange: m1, Green: m2
# plt.text(-1, 0.1, '$m_1$', color='orange', fontsize=12, ha='center')
# plt.text(1, 0.1, '$m_2$', color='green', fontsize=12, ha='center')

# Step 7: Customize plot
plt.title("Gravitational Field Due to Two Point Masses")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True)
plt.colorbar(label='Gravitational Field Magnitude')
plt.show()
