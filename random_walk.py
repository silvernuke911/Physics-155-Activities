import numpy as np
import matplotlib.pyplot as plt

N = 200
output = np.zeros(N)
output[0] = 0
for i in range(1, N):
    if np.random.rand() >= 0.5:
        output[i] = 1
    else:
        output[i] = -1

print(output)
out_sum = np.cumsum(output)
print(out_sum)

# Correct the arguments in plt.stairs
plt.grid()
plt.stairs(out_sum, range(N+1),linewidth=2,zorder = 2)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the random walk
N = 2000  # Number of steps

# Generate random steps: 1 or -1 with equal probability
steps = np.where(np.random.rand(N) >= 0.5, 1, -1)

# Calculate the random walk by taking the cumulative sum of the steps
random_walk = np.cumsum(steps)

# Plot the random walk
plt.figure(figsize=(10, 6))
plt.plot(random_walk, linewidth=2)
plt.xlabel("Step Number", fontsize=12)
plt.ylabel("Position", fontsize=12)
plt.grid(True)
plt.show()

for i in range(50):
    steps = np.where(np.random.rand(1000) >= 0.5, 1, -1)
    random_walk = np.cumsum(steps)
    plt.plot(random_walk, linewidth=2)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the random walk
N = 10000  # Number of steps

# Generate random steps: -1, 0, or 1 for both x and y directions
# Each step is chosen randomly from {-1, 0, 1}
steps_x = np.random.choice([-1, 1], N)
steps_y = np.random.choice([-1, 1], N)

# Calculate the cumulative sum to get the positions in both x and y directions
x_positions = np.cumsum(steps_x)
y_positions = np.cumsum(steps_y)

# Plot the 2D random walk
plt.figure(figsize=(8, 8))
plt.plot(x_positions, y_positions, linewidth=2)
plt.title("2D Random Walk", fontsize=16)
plt.xlabel("X Position", fontsize=12)
plt.ylabel("Y Position", fontsize=12)
plt.grid(True)

# Mark the starting and ending points
plt.plot(0, 0, 'ro', label="Start")  # Start point (0, 0)
plt.plot(x_positions[-1], y_positions[-1], 'bo', label="End")  # End point
plt.legend()
plt.gca().set_aspect('equal')
plt.show()



