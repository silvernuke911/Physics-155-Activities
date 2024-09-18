import numpy as np
import matplotlib.pyplot as plt

# Aesthetic decision
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

# 2.6

# Bad quadratic formula implementation

def quadratic_formula_bad(a,b,c):
    discriminant = b**2 - 4 * a * c
    if discriminant < 0 :
        return 'Zeros does not exist'
    else:
        x1 = (-b - np.sqrt(discriminant)) / (2*a)
        x2 = (-b + np.sqrt(discriminant)) / (2*a)
        return x1,x2

# Good quadratic formula implementation

def quadratic_formula_gud(a,b,c):
    discriminant = b**2 - 4 * a * c
    if discriminant < 0 :
        return 'Zeros does not exist'
    else:
        x1 = (-b - np.sqrt(discriminant)) / (2*a)
        x2 = (c/a) / x1
        return x1,x2

# Defining constants and b variables
a,c = 1,1
b = np.logspace(7,8,20)

# Defining solution bins
neg_root_bad = np.zeros_like(b)
pos_root_bad = np.zeros_like(b)
neg_root_gud = np.zeros_like(b)
pos_root_gud = np.zeros_like(b)

# Filling thoses solution bins
for i,b_val in enumerate(b):
    bad_sol = quadratic_formula_bad(a,b_val,c)
    gud_sol = quadratic_formula_gud(a,b_val,c)
    neg_root_bad[i] = bad_sol[0]
    pos_root_bad[i] = bad_sol[1]
    neg_root_gud[i] = gud_sol[0]
    pos_root_gud[i] = gud_sol[1]

# Taking absolute value
pos_root_gud = abs(pos_root_gud)
pos_root_bad = abs(pos_root_bad)

# Plotting
xtick = np.linspace(1e7,1e8,10)
ytick = np.linspace(1e-7,1e-8,10)
latex_font()
fig, ax = plt.subplots()
plt.title('Comparison of computation methods for the quadratic formula')
plt.xlabel(r'$b$')
plt.ylabel(r'$|x_+|$')
ax.plot(b,pos_root_gud, marker = '.', color = 'b', label = 'Good method')
ax.plot(b,pos_root_bad, marker = '.', color = 'r', label = 'Bad method')
plt.xscale('log')
plt.yscale('log')
plt.xticks(xtick)
plt.yticks(ytick)
ax.set_aspect(0.6, adjustable='box')
plt.legend()
plt.grid()
plt.show()