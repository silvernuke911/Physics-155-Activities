import numpy as np
import matplotlib.pyplot as plt

def latex_font():
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

# Constants
N = 2000
V0 = -2.2 # MeV
a = 2 #fm
mu = 469.5 # Reduced mass (MeV/c^2)
extent = 3
E = -2.2
x = np.linspace(-extent * a, extent * a, N)
dx = x[1]-x[0]

# Potential function
# def potential_function(x):
#     return V0 * (np.abs(x) <= a)

def potential_function(x):
    output = np.zeros_like(x)
    for i,x_i in enumerate(x):
        if x_i < -2 :
            output[i] = 10000
        if x_i > -2 and x_i <= 0 :
            output[i] = V0+0.4
        if x_i > 0 and x_i < 2 :
            output[i] = V0
        if x_i > 2 :
            output[i] = 10000
    # output = 0.1* x**2 - 3
    return output

# Hamiltonian
main_diag = -2 * np.ones(N)
off_diag = np.ones(N-1)
derivative_matrix = (np.diag(main_diag) + np.diag(off_diag,-1) + np.diag(off_diag,1))/dx**2
potential_matrix = np.diag(potential_function(x))
hamiltonian_operator = (- 1/(2* mu)) * derivative_matrix + potential_matrix

# Solving for the eigenfunction
eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_operator)

# Find the ground state (first eigenvalue)
n = 50
ground_state_energy = eigenvalues[n]
ground_state_wavefunction = eigenvectors[:, n]

latex_font()
V = potential_function(x)
plt.figure(figsize=(10, 8))
plt.plot(x, V, label='Potential Well $V(x)$', color = 'b')
plt.plot(x, ground_state_wavefunction * 10, color = 'r', label = r'Wave function ($\psi(x)$)')
plt.axhline(y=ground_state_energy, color='green', linestyle='--', label=f'State Energy: {ground_state_energy:.3f} MeV')
plt.xlabel(r'$x$ (fm)')
plt.xlim([min(x),max(x)])
plt.ylabel('Energy (MeV)')
ax = plt.gca()
ax.set_aspect(2.5, adjustable='box')
ax.set_axisbelow(True)
plt.grid()
plt.legend()
plt.show()
