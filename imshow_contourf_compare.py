import numpy as np
import matplotlib.pyplot as plt
import time

# Create a large random 2D array
data = np.random.rand(1000, 1000)

# Benchmark imshow
start_time = time.time()
plt.imshow(data, cmap='viridis')
plt.colorbar()
plt.title('imshow')
plt.show()
print(f'imshow time: {time.time() - start_time:.4f} seconds')

# # Benchmark contourf
# start_time = time.time()
# plt.contourf(data, levels=20, cmap='viridis')
# plt.colorbar()
# plt.title('contourf')
# plt.show()
# print(f'contourf time: {time.time() - start_time:.4f} seconds')


# Constants
G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
M = 5.9722e24   # mass of Earth in kg
R = 6.371e6    # radius of Earth in meters

# Escape velocity formula
v_esc = np.sqrt((2 * G * M) / R)
print(v_esc)

rho = 5.5e3
period = np.sqrt((3*np.pi) / (G * rho))
print(period)
def convert_seconds(seconds):
    # Calculate hours
    hours = int(seconds // 3600)
    # Calculate minutes
    minutes = int((seconds % 3600) // 60)
    # Calculate remaining seconds
    secs = seconds % 60
    return hours, minutes, secs
print(convert_seconds(period))