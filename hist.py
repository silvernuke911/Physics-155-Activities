import matplotlib.pyplot as plt
import numpy as np

# Initial normal distribution plot
sigma = 1
mu = 0
sample = np.random.normal(mu, sigma, 1000000)
n_bin = 200

count, bins, ignored = plt.hist(sample, n_bin, density=True, color='gray')
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
plt.show()

# Scatter plot of normal distribution
N = 5000
x_rand = np.random.normal(mu, sigma, N)
y_rand = np.random.normal(mu, sigma, N)
plt.scatter(x_rand, y_rand, color='r', marker='.', zorder=3)
plt.grid()
plt.gca().set_aspect('equal')
plt.show()

# Heatmap of distribution
N_smal = 100
x_coord = np.linspace(-5, 5, N_smal + 1)  # Added +1 for proper binning
y_coord = np.linspace(-5, 5, N_smal + 1)
heat_map = np.zeros([N_smal, N_smal])

# Update heat_map by checking points within bins
for k in range(len(x_rand)):
    x_bin = np.digitize(x_rand[k], x_coord) - 1  # Find the bin index for x_rand
    y_bin = np.digitize(y_rand[k], y_coord) - 1  # Find the bin index for y_rand

    if 0 <= x_bin < N_smal and 0 <= y_bin < N_smal:
        heat_map[x_bin][y_bin] += 1

plt.imshow(heat_map, cmap = 'hot', extent = [-5,5,-5,5], origin = 'lower')
plt.xticks(np.arange(-5,6))
plt.yticks(np.arange(-5,6))
plt.show()