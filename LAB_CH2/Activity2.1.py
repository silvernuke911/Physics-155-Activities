import numpy as np
import matplotlib.pyplot as plt

#Aesthetic decision
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
# 10^7 to 10^8
def quadratic_formula_bad(a,b,c):
    discriminant = b**2 - 4 * a * c
    if discriminant < 0 :
        return 'Zeros does not exist'
    else:
        x1 = (-b - np.sqrt(discriminant)) / (2*a)
        x2 = (-b + np.sqrt(discriminant)) / (2*a)
        return x1,x2
    
def quadratic_formula_gud(a,b,c):
    discriminant = b**2 - 4 * a * c
    if discriminant < 0 :
        return 'Zeros does not exist'
    else:
        x1 = (-b - np.sqrt(discriminant)) / (2*a)
        x2 = (c/a) / x1
        return x1,x2

a,c = 1,1
b = np.logspace(7,8,20)

neg_root_bad = np.zeros_like(b)
pos_root_bad = np.zeros_like(b)
neg_root_gud = np.zeros_like(b)
pos_root_gud = np.zeros_like(b)

for i,b_val in enumerate(b):
    bad_sol = quadratic_formula_bad(a,b_val,c)
    gud_sol = quadratic_formula_gud(a,b_val,c)
    neg_root_bad[i] = bad_sol[0]
    pos_root_bad[i] = bad_sol[1]
    neg_root_gud[i] = gud_sol[0]
    pos_root_gud[i] = gud_sol[1]

pos_root_gud = abs(pos_root_gud)
pos_root_bad = abs(pos_root_bad)

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
ax.set_aspect(0.8, adjustable='box')
plt.legend()
plt.grid()
plt.show()

# 2.18
# # create original function, perform a 
# # perform b, and how long does it take to run

def sqwave_true(x,frequency=0.25,amplitude=1,phase=0,offset=0):
    y=amplitude*np.sin(2*np.pi*frequency*x+phase)+offset
    output = np.zeros_like(x)
    for i,y_val in enumerate(y): 
        if y_val >= 0:
            output[i] = amplitude
        if y_val <0:
            output[i] = -amplitude
    return output
    
def sqwave_approx(x,niters):
    output = np.zeros_like(x)
    def sin_sum(x,max_iter):
        out = 0
        for i in range(max_iter):
            out += np.sin((2*i + 1) * x) / (2*i + 1)
        return out
    for i,x_val in enumerate(x):
        output[i] = (2/np.pi) * sin_sum(x_val,niters)
    return output

max_side,min_side = -4*np.pi, 4*np.pi
x = np.linspace(-4*np.pi, 4*np.pi, 1000)
y_true = sqwave_true(x,0.5/np.pi,0.5,0,0)
y_approx = sqwave_approx(x,150)


plt.subplot(2, 1, 1)
plt.grid()
xtick = np.arange(max_side,min_side+np.pi, np.pi)
plt.xlim([max_side,min_side])
plt.xticks(xtick,[r'-4$\pi$',r'-3$\pi$',r'-2$\pi$',r'-$\pi$',r'0',r'$\pi$',r'$2\pi$',r'$3\pi$',r'$4\pi$'])
plt.plot(x,y_true, color = 'r')
plt.title('Analytic piecewise function')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.subplot(2, 1, 2)
plt.grid()
xtick = np.arange(max_side,min_side+np.pi, np.pi)
plt.xlim([max_side,min_side])
plt.xticks(xtick,[r'-4$\pi$',r'-3$\pi$',r'-2$\pi$',r'-$\pi$',r'0',r'$\pi$',r'$2\pi$',r'$3\pi$',r'$4\pi$'])
plt.plot(x,y_approx, color = 'b')
plt.title('Approximate piecewise function (summation)')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.show()

    
