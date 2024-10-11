import numpy as np
import matplotlib.pyplot as plt

num_walks = 1
n_steps = 100000
grid_size = 1000
grid = np.zeros((grid_size,grid_size))
for _ in range(num_walks):
    x_steps = np.zeros((n_steps))
    y_steps = np.zeros((n_steps))
    for i in range(1, n_steps):
        direction = np.random.choice([0, 1, 2, 3])
        if   direction == 0:
            x_steps[i] = -1
        elif direction == 1:
            x_steps[i] =  1
        elif direction == 2:
            y_steps[i] = -1
        else:
            y_steps[i] =  1
    x_path = np.cumsum(x_steps) + grid_size // 2
    y_path = np.cumsum(y_steps) + grid_size // 2
    x_path = np.clip(x_path, 0, grid_size - 1).astype(int)
    y_path = np.clip(y_path, 0, grid_size - 1).astype(int)
    for i in range(n_steps):
        grid[x_path[i], y_path[i]] += 1


max_value = np.max(grid)
truncated_max = max_value // 1
plt.imshow(grid.T, cmap='inferno', vmin=0, vmax=truncated_max)
plt.show()

def surface_plot():
    # Prepare 3D data
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    x, y = np.meshgrid(x, y)
    z = grid.T  # Transpose so that orientation matches the grid layout

    # Create 3D surface plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface with a colormap
    surf = ax.plot_surface(x, y, z, cmap='inferno', edgecolor='none')

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Visits')

    # Labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Visits')
    ax.set_title('3D Surface Plot of Random Walk')

    plt.show()

def bar_plot_3d():
    # Set up 3D bar plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare the grid indices and heights for the bars
    x_pos, y_pos = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)

    # Heights of the bars
    dz = grid.flatten()

    # Bar size
    dx = dy = np.ones_like(z_pos)

    # Color map based on the heights (visits count)
    colors = plt.cm.inferno(dz / np.max(dz))

    # Create 3D bar plot
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, zsort='average')

    # Labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Visits')
    ax.set_title('3D Bar Plot of Random Walk')

    # Show plot
    plt.show()