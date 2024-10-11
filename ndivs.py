import numpy as np
import matplotlib.pyplot as plt

# Define the grid
dx = dy = 0.001
a = 1
x = np.arange(-a, a, dx)
y = np.arange(-a, a, dy)
X, Y = np.meshgrid(x, y)

minima = 0
maxima = 1000
# Calculate R and clip it at 100
R = 1 / (X**2 + Y**2)
R = np.clip(R,minima,maxima)  # Limit R to a maximum of 100
print(R)
# Create the contour plot, capping the colorbar at 100
plt.contourf(X, Y, R, levels = 100, cmap='inferno', vmax=maxima, vmin=minima)
plt.colorbar()
plt.title('Contour Plot of $R = 1 / (X^2 + Y^2)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

plt.imshow(R,cmap = 'hot')
plt.show()

