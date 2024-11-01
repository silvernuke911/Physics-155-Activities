import numpy as np
import matplotlib.pyplot as plt

def latex_font(): # Aesthetic choice
    import matplotlib as mpl
    import matplotlib.font_manager as font_manager
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.unicode_minus': False,
        'axes.formatter.use_mathtext': True,
        'font.size': 12
    })
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
latex_font()

G = 6.6743e-11  # gravitational constant in m^3 kg^-1 s^-2
M = 5.9722e24   # mass of shell (for example, Earth's mass) in kg
m = 1           # mass of particle in kg
R1 = 6.371e6    # inner radius of shell (example: Earth's radius) in meters
R2 = 8.371e6    # outer radius (same as inner for simplicity here)

dr = 0.01
r = np.linspace(0,3*R2,10000)


F = np.zeros_like(r)
F[r > R2] = -G * M * m / r[r > R2]**2

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(r, F, label=r'Gravitational Force $F(r)$',linewidth = 2, color = 'b')
plt.axvline(x=R1, color='r', linestyle='--', label=r'Inner Radius $R_1$')
plt.axvline(x=R2, color='g', linestyle='--', label=r'Outer Radius $R_2$')

plt.xlabel('$r$ (m)')
plt.ylabel('$F(r)$ (N)')
plt.xlim(0,3*R2)
plt.legend(fontsize = 8)
plt.grid(True)
plt.savefig('sphereshell.png')
plt.show()