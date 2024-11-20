import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# LaTeX font, as an aesthetic choice
def latex_font(): 
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 12
    })
latex_font()

# Define polar coordinates for the Riemann sheets
r = np.linspace(0.01, 2, 100) 
theta = np.linspace(0, 2 * np.pi, 100)

# Create a meshgrid for r and theta
R, Theta = np.meshgrid(r, theta)

# Compute the real and imaginary parts of each branch of f(z) = z^(1/4)
Z_real = R ** (1/4) * np.cos(Theta / 4)  # Real part
Z_imag = R ** (1/4) * np.sin(Theta / 4)  # Imaginary part

# Initialize the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot four sheets, each corresponding to a different branch,
#  each branch with different color
colors = ['blue', 'green', 'orange', 'purple']
for i in range(4):
    sheet_theta = Theta + i * 2 * np.pi
    Z_real = R ** (1/4) * np.cos(sheet_theta / 4)
    Z_imag = R ** (1/4) * np.sin(sheet_theta / 4)

    ax.plot_surface(Z_real, Z_imag, sheet_theta, color=colors[i], alpha=0.6, edgecolor='none')

# Labels and plot adjustments
ax.set_xlabel('Real part of $f(z)$')
ax.set_ylabel('Imaginary part of $f(z)$')
ax.set_zlabel('Argument of $z$')
ax.set_title(r"\textbf{Riemann Surface for} $f(z) = z^{1/4}$")

plt.savefig('riemann_surface.png', format="png", dpi=300)
plt.show()
