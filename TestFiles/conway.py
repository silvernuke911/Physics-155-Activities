import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize the grid with random 0s (dead) and 1s (alive)
def initialize_grid(size):
    grid = np.random.choice([0, 1], size * size, p=[0.8, 0.2]).reshape(size, size)
    return grid

# Function to count the number of live neighbors for each cell
def count_neighbors(grid):
    neighbors = (
        np.roll(np.roll(grid, 1, 0), 1, 1) + np.roll(np.roll(grid, 1, 0), -1, 1) +
        np.roll(np.roll(grid, -1, 0), 1, 1) + np.roll(np.roll(grid, -1, 0), -1, 1) +
        np.roll(grid, 1, 0) + np.roll(grid, -1, 0) +
        np.roll(grid, 1, 1) + np.roll(grid, -1, 1)
    )
    return neighbors

# Function to apply Conway's Game of Life rules
def update_grid(grid):
    neighbors = count_neighbors(grid)
    # Apply the rules of the Game of Life
    new_grid = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 3)), 0, grid)  # Live cells with <2 or >3 neighbors die
    new_grid = np.where((grid == 0) & (neighbors == 3), 1, new_grid)  # Dead cells with exactly 3 neighbors become alive
    return new_grid

# Function to animate the grid
def animate(i, grid, img):
    new_grid = update_grid(grid)
    img.set_data(new_grid)
    grid[:] = new_grid[:]  # Update the grid in place
    return img,

# Main function to run the Game of Life simulation
def game_of_life(size=50, iterations=100, interval=200):
    grid = initialize_grid(size)
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest', cmap='binary')
    
    ani = animation.FuncAnimation(fig, animate, fargs=(grid, img), frames=iterations, interval=interval, save_count=50)
    
    plt.show()

# Run the Game of Life
game_of_life(size=50, iterations=100, interval=200)
