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

# Benchmark contourf
start_time = time.time()
plt.contourf(data, levels=20, cmap='viridis')
plt.colorbar()
plt.title('contourf')
plt.show()
print(f'contourf time: {time.time() - start_time:.4f} seconds')