import numpy as np
import matplotlib.pyplot as plt

theta = np.arange(-2*np.pi,2*np.pi,0.01)
real = np.cos(theta)
imaginary = np.sin(theta)

plt.plot(real, imaginary)
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

real = np.cos((theta + 2*np.pi)/2)
imaginary = np.sin((theta + 2*np.pi)/2)

plt.plot(real, imaginary)
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()