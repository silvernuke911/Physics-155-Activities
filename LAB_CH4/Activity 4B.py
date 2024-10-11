import matplotlib.pyplot as plt
import numpy as np
# Setting constants
R = 1
V_0 = 5
# Solver matrix which returns the voltage drop vector x, and the coefficient matrix A
def resistor_solve(n, R = 1,V0 = 5):
    def generate_A(n):
        output = np.zeros((n, 2 * n + 1))
        for i in range(n):
            if 2 * i < 2 * n:
                output[i, 2 * i] = 1  
            if 2 * i + 1 < 2 * n:
                output[i, 2 * i + 1] = 1  
            if 2 * i + 3 < 2 * n:
                output[i, 2 * i + 3] = -1 
            if 2 * i + 4 < 2 * n:
                output[i, 2 * i + 4] = -1 
        output =  np.roll(output, -1)
        if len(output)<2:
            return np.array([])
        else :
            output[-1][-1]=-1
            output[-2][-2]=-1
        return output.T
    A = generate_A(n)
    K = (1/R) * np.eye(2 * n + 1)
    V = np.zeros((2*n + 1))
    V[0] = V0
    V[1] = V0
    V = np.array([V]).T
    X = np.linalg.solve(A.T@K@A,A.T@K@V)
    return A, X
A, x = resistor_solve(6)
print(A)
print(x)


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

N = 30
index = range(1,N+1)
volta = resistor_solve(N)[1].T
plt.grid()
plt.scatter(index, volta, marker = '.', color = 'r', zorder = 2)
plt.ylim([0,5])
plt.xlim([0,N+0.5])
plt.xlabel('Junction index $x_i$')
plt.ylabel('Voltage at junction (V)')
plt.savefig("voltage_drop_plot.png", dpi=300)
plt.show()



