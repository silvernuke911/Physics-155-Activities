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

latex_font()
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