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

def small_circle(center,radius):
    theta = np.linspace(0, 2*np.pi, 360)
    x = center[0] + 0.3 * radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x,y


def U_func(x, U_0, a):
    U = U_0 * (2 * (x/a)**2 - (x/a)**4)
    return U

lims = 2
x = np.linspace(-lims,lims,200)
U_0 = 1
a = 1

latex_font()
plt.xlim([-lims,lims])
plt.ylim([-8,2])
# plt.title('Potential Energy function for a particle')
plt.xlabel(r'Radius $r$ $(a)$')
plt.ylabel(r'Potential Energy $U$ $(U_0)$')
plt.plot(x,U_func(x,U_0,a), color = 'blue')
plt.plot(small_circle([1,1],0.2)[0],small_circle([1,1],0.2)[1], color = 'red')
plt.plot(small_circle([-1,1],0.2)[0],small_circle([-1,1],0.2)[1], color = 'red')
plt.plot(small_circle([0,0],0.2)[0],small_circle([0,0],0.2)[1], color = 'green')
ax = plt.gca()
ax.set_aspect(0.3, adjustable='box')
ax.set_axisbelow(True)
plt.grid(True)
plt.show()