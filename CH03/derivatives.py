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

def func(x):
    return x*x*x
def dev_func(x):
    return 3*x**2

dx = 0.05
x = np.arange(-2,2,dx)
y_fwd = forward_difference(func,x,dx)
y_bwd = backward_difference(func,x,dx)
y_cnt = central_difference(func,x,dx)
y_true = dev_func(x)

plt.plot(x,y_fwd,color = 'r')
plt.plot(x,y_bwd,color = 'b')
plt.plot(x,y_cnt,color = 'g')
plt.plot(x,y_true,color = 'c')
plt.grid()
plt.show()

