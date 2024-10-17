import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# Define the reimann zeta function for reals
def zeta0(s, max_iter = 10000, tol = 1e-16):
    output = np.zeros_like(s)
    for i, s_i in enumerate(s):
        zeta_sum = 0
        for n in range(1,max_iter):
            zeta_sum += 1 / n**s_i
            if 1 / n**s_i < tol:
                break
        output[i] = zeta_sum
    return output
def zeta(s, max_iter=10000, tol=1e-16):
    n_values = np.arange(1, max_iter)
    output = np.zeros_like(s, dtype=float)
    for i, s_i in enumerate(s):
        terms = 1 / n_values**s_i
        zeta_sum = np.sum(terms[terms >= tol])
        output[i] = zeta_sum
    return output

x = np.arange(1,20,0.01)
y = zeta(x)

plt.plot(x,y,'b')
plt.plot(x,1/(x-1)+1,'r')
plt.grid()
plt.xlim(1,10)
plt.ylim(0,10)
plt.yticks(range(10+1))
plt.show()
