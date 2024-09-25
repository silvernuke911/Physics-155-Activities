import numpy as np
import matplotlib.pyplot as plt

def forward_difference(f,x,dx):
    output = np.zeros_like(x)
    for i in range(len(x)):
        if i == len(x)-1:
            output[i] = (f(x[i]) - f(x[i-1])) / dx
        elif i < len(x)-1:
            output[i] = (f(x[i+1]) - f(x[i])) / dx
    return output

def backward_difference(f,x,dx):
    output = np.zeros_like(x)
    for i in range(len(x)):
        if i == 0:
            output[i] = (f(x[i+1]) - f(x[i])) / dx
        if i > 0:
            output[i] = (f(x[i]) - f(x[i-1])) / dx
    return output

def central_difference(f,x,dx):
    output = np.zeros_like(x)
    for i in range(len(x)):
        if i==0:
            output[i] = (f(x[i+1]) - f(x[i])) / dx
        elif i==len(x)-1:
            output[i] = (f(x[i]) - f(x[i-1])) / dx 
        else:
            output[i] = (f(x[i+1]) - f(x[i-1])) / (2 * dx)
    return output

def double_derivative(f,x,dx):
    output = np.zeros_like(x)
    for i in range(len(x)):
        if i == 0:
            output[i] = (f(x[i+2]) - 2*f(x[i+1]) + f(x[i])) / dx**2
        if i == len(x):
            output[i] = (f(x[i-2]) - 2*f(x[i-1]) + f(x[i])) / dx**2
        else:
            output[i] = (f(x[i+1]) - 2*f(x[i]) + f(x[i-1])) / dx**2
    return output

def single_derivative_operator(x,dx):
    n = len(x)
    dydx_mat = np.zeros(n) + np.diag(np.ones(n-1),1)-np.diag(np.ones(n-1),-1)
    dydx_mat[0,0], dydx_mat[-1,-1] = -2, 2
    dydx_mat[0,1], dydx_mat[-1,-2] = 2, -2
    return dydx_mat / (2 * dx)

def func(x):
    return np.sin(x)+x
def dev_func(x):
    return np.cos(x)+1

dx = 0.001
x = np.arange(-2,2,dx)
y_fwd = forward_difference(func,x,dx)
y_bwd = backward_difference(func,x,dx)
y_cnt = central_difference(func,x,dx)
y_grad = np.gradient(func(x),x)
y_true = dev_func(x)
y_mat = single_derivative_operator(x,dx)@func(x).T

plt.plot(x,y_fwd,color = 'r')
plt.plot(x,y_bwd,color = 'b')
plt.plot(x,y_cnt,color = 'g')
plt.plot(x,y_grad,color = 'm')
plt.plot(x,y_true,color = 'c')
plt.plot(x,y_mat,color = 'y')
plt.grid()
plt.show()

