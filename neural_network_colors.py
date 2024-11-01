import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activation_calculation(in_put, weight_matrix, bias_matrix):
    return sigmoid(weight_matrix @ in_put - bias_matrix)

color = [255,248,255]
color_normalized = np.array([color]).T / 255

print(color_normalized)
weight_matrix_L1 = np.random.rand(10,3)
print(weight_matrix_L1)
bias_matrix_L1 = np.random.rand(10,1)
print(bias_matrix_L1)

result_matrix = activation_calculation(color_normalized,weight_matrix_L1, bias_matrix_L1)
print(result_matrix)

weight_matrix_L2 = np.random.rand(10,10)
print(weight_matrix_L2)

plt.imshow(weight_matrix_L2)
plt.show()


import numpy as np

# Define a function to map RGB values to color labels
def rgb_to_color_label(rgb):
    r, g, b = rgb
    
    if r > 200 and g < 100 and b < 100:
        return 'red'
    elif r > 200 and g > 150 and b < 50:
        return 'orange'
    elif r > 200 and g > 200 and b < 100:
        return 'yellow'
    elif r < 100 and g > 200 and b < 100:
        return 'green'
    elif r < 100 and g < 100 and b > 200:
        return 'blue'
    elif r > 100 and g < 100 and b > 200:
        return 'violet'
    elif r > 200 and g > 200 and b > 200:
        return 'white'
    elif r < 50 and g < 50 and b < 50:
        return 'black'
    elif abs(r - g) < 20 and abs(g - b) < 20 and r < 200:
        return 'gray'
    elif r > 100 and g < 100 and b < 50:
        return 'brown'
    elif r < 100 and g > 200 and b > 200:
        return 'cyan'
    elif r > 200 and g < 100 and b > 200:
        return 'magenta'
    else:
        return 'unknown'

# Generate 1000 random RGB values and assign labels
data = []
for _ in range(1000):
    rgb = list(np.random.randint(0, 256, size=3))  # Generate random RGB values
    color_label = rgb_to_color_label(rgb)          # Map RGB to a color label
    data.append([rgb, color_label])                # Append in the required format

# Print a few examples from the list
print(data[:100])  # Print the first 10 entries
