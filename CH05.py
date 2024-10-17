# Library imports
import numpy as np
import matplotlib.pyplot as plt

# Bisection function provided by Ma'am Lim
def bisection_method(f, x_left=-0.01, x_right=2, delta=1e-8):
    root = None
    sigma = (x_right - x_left)/2
    def get_midpoint(x_left, x_right):
        return (x_left + x_right)/2
    def bisect_once(x_left, x_right, f):
        x_mid = get_midpoint(x_left, x_right)
        if f(x_left) * f(x_mid) <= 0:
            x_right = x_mid
        else:
            x_left = x_mid
        return x_left, x_right
    def check_bisect(x0, x1, f):
        if (f(x0) * f(x1)) <= 0:
            return True
        else:
            return False
    if check_bisect(x_left, x_right, f):
        while np.abs(sigma/get_midpoint(x_left, x_right)) > delta:
            x_left, x_right = bisect_once(x_left, x_right, f)
            root = get_midpoint(x_left, x_right)
            sigma = (x_right - x_left)/2
    return root, sigma

def newtons_method(f,x0, max_iter = 1000, tol = 1e-10, h = 1e-10):
    def derivative(f, x, h = 1e-10):
        return (f(x + h) - f(x - h)) / (2 * h)
    x = x0
    for _ in range(max_iter):
        f_val = f(x)
        df_val = derivative(f, x, h)
        if df_val == 0:
            print(f"Derivative is zero at x = {x}. Stopping iteration to avoid division by zero.")
            return np.nan
        x_new = x - f_val / df_val
        # Check if the difference is within tolerance
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    print(f"Exceeded maximum iterations at x = {x}")
    return np.nan
        

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

# Plotting the functions
dt = 0.2
x = np.linspace(-0.5, 2, 1000)
for t_pick in np.arange(0.2, 2+dt, dt):
    m_func = lambda m: m - np.tanh(m/t_pick)
    plt.plot(x, m_func(x), label=f"{t_pick:.2f}")

plt.axhline(y=0,color = 'k', zorder = 3)
plt.legend(fontsize=9)
plt.grid()
plt.xlim([-.5,2])
plt.xlabel('$m$')
plt.ylabel('$y$')
plt.title(r'Plot of $y = m - \tanh\left(\frac{m}{t}\right)$ for varying values of $t$')
plt.show()

# # Defining the function
# def func(m, t):
#     return m - np.tanh( m / t )

# Computing the roots using the bisection function
dt = 0.001
t_list = np.arange(0 + dt, 2, dt)
m_list = np.zeros_like(t_list)

# for i, t in enumerate(t_list):
#     m_list[i] = bisection_method(lambda m: m - np.tanh( m / t ), 0.01,2)[0]
#     if np.isnan(m_list[i]) :
#         m_list[i] = bisection_method(lambda m: m - np.tanh( m / t ), -0.01,1)[0]

for i, t in enumerate(t_list):
    m_list[i] = newtons_method(lambda m: m - np.tanh( m / t ), 2)
    if np.isnan(m_list[i]) :
        m_list[i] = 0

# Plotting the roots
plt.plot(t_list,m_list, color = 'r', zorder = 2)
plt.grid()
plt.xlim([0,2])
plt.xlabel('$t$')
plt.ylabel('$m$')
plt.title(r'Values of $m$ that satisfies $m = \tanh\left(\frac{m}{t}\right)$')
plt.show()

dx = 0.001
x = np.arange(0,12,dx)
plt.plot(x,np.tan(x),color = 'b',zorder = 2)
plt.plot(x,np.sqrt((12/x)**2 - 1), color = 'r')
plt.grid()
plt.xlim([0,12])
plt.ylim([0,50])
plt.xlabel('$z$')
plt.show()

plt.plot(x,np.tan(x) - np.sqrt((12/x)**2 - 1),color = 'r',zorder = 2)
plt.grid()
plt.xlim([0,12])
plt.ylim([-50,50])
plt.xlabel('$z$')
plt.show()
# Problem was skipped :<

# Define the function phi(x, y)
def phi(x, y):
    return x**2 + 2*x + y**4 - 2*y**2 + y 

# Define the constraint function g(x, y)
def g(x, y):
    return 5*x**2 - 3*y**3 - 7

# Create a grid of x and y values
x_vals = np.linspace(-4, 2, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute phi(x, y) and g(x, y) for each point on the grid
Z = phi(X, Y)
G = g(X, Y)

# Plot the contour of phi(x, y)
plt.figure(figsize=(6, 6))
contourf = plt.contourf(X, Y, Z, levels=100, cmap='inferno')

# Add the constraint line g(x, y) = 0
plt.contour(X, Y, G, levels=[0], colors='b', linewidths=2 )

# Adding minima (Can be solved for with the script below, value is (-1, -1.107)
plt.scatter(-1,-1.107, color = 'r', label = 'mimima')

# Add colorbar and labels for the filled contour plot, and plot the resulting thingy
plt.colorbar(contourf)
plt.legend(fontsize = 8)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(r'Contour plot of $\phi(x, y)$ with constraint $g(x, y) = 0$')
plt.show()

import scipy as sp

def phi(x_vec):
    x = x_vec[0]
    y = x_vec[1]
    return x**2 + 2*x + y**4 - 2*y**2 + y

def g(x_vec):
    x = x_vec[0]
    y = x_vec[1]
    return 5*x**2 - 3*y**3 - 7
    
output = sp.optimize.minimize(phi, method="Nelder-Mead", x0=[0,0])
print(output)
# output is (-1,-1.107)
