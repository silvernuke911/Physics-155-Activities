import numpy as np
import matplotlib.pyplot as plt


dx = 0.01
x = np.arange(0,10,dx)

def func(x):
    return x**2 + np.sin(x)


def single_derivative_operator(x,dx):
    n = len(x)
    side_diag = np.ones(n-1)
    big_diag = np.zeros(n) + np.diag(side_diag,1)-np.diag(side_diag,-1)
    big_diag[0,0], big_diag[-1,-1] = -2, 2
    big_diag[0,1], big_diag[-1,-2] = 2, -2
    return big_diag / (2 * dx)

print(single_derivative_operator(x,1))

def double_derivative_operator(x,dx):
    n = len(x)
    main_diag = -2*np.diag(np.ones(n))
    side_diag = np.ones(n-1)
    big_diag = main_diag+np.diag(side_diag,1)+np.diag(side_diag,-1)
    big_diag[0,0], big_diag[0,1], big_diag[0,2] = 1, -2, 1
    big_diag[-1,-1], big_diag[-1,-2], big_diag[-1,-3] = 1, -2, 1
    return big_diag / dx**2

y_mat = single_derivative_operator(x,dx) @ func(x).T

plt.plot(x,func(x), color = 'm')
plt.plot(x,y_mat, color = 'b')
plt.plot(x,np.gradient(func(x),x), color = 'r')
plt.show()
