import numpy as np
import matplotlib.pyplot as plt
# Parameters
num_walks = 10000  # Number of random walks
N = 1000  # Number of steps in each random walk
grid_size = int(N/2)  # Define grid size to cover all possible positions
heat_map_grid = np.zeros((N,grid_size))
for i in range(num_walks):
    output = np.zeros(N)
    output[0] = 0
    for i in range(1, N):
        if np.random.rand() >= 0.5:
            output[i] = 1
        else:
            output[i] = -1
    out_sum = np.cumsum(output)
    out_sum += int(grid_size/2)
    for i in range(N):
        num = int(out_sum[i])
        heat_map_grid[i][num] = heat_map_grid[i][num]+1
print(heat_map_grid.T)

max_value = np.max(heat_map_grid)
truncated_max = max_value / 10

plt.imshow(heat_map_grid.T, cmap='inferno', vmin=0, vmax=truncated_max)
plt.show()

histo = heat_map_grid[400,:]
plt.bar(range(len(histo)),histo)
plt.xlim(200,300)
plt.show()
