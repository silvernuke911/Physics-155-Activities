import numpy as np 
import matplotlib.pyplot as plt 

def forward_euler(f, x, y_0, dx):
    y = np.zeros_like(x) 
    y[0] = y_0
    for i in range(len(x) - 1):
        y[i+1] = y[i] + f(x[i], y[i]) * dx  
    return y

def mid_point_method(f, x, y_0, dx):
    y = np.zeros_like(x) 
    y[0] = y_0
    for i in range(len(x)-1):
        y[i+1]=y[i] + dx * f(x[i] + dx/2, y[i] + (dx/2)*f(x[i],y[i]))
    return y

def rk4(f, x, y_0, dx):
    y = np.zeros_like(x) 
    y[0] = y_0
    for i in range(len(x)-1):
        k1 = f(x[i],y[i])
        k2 = f(x[i] + dx/2, y[i] + dx*k1/2)
        k3 = f(x[i] + dx/2, y[i] + dx*k2/2)
        k4 = f(x[i] + dx, y[i] + dx*k3)
        y[i + 1] = y[i] + (dx/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y

def rk2(f, x, y_0, dx):
    y = np.zeros_like(x) 
    y[0] = y_0
    for i in range(len(x)-1):
        k1 = f(x[i],y[i])
        k2 = f(x[i] + dx/2, y[i] + dx*k1/2)
        y[i + 1] = y[i] + (dx/6)*(k1 + 2*k2)
    return y

def func(x,y):
    return x*y

dx = 0.1
x = np.arange(0,10,dx)
y_0 = 1

y_fwd = forward_euler(func,x,y_0,dx)
y_mdp = mid_point_method(func,x,y_0,dx)
y_rk4 = rk4(func,x,y_0,dx)

plt.plot(x,y_fwd,color ='b',marker ='o')
plt.plot(x,np.exp(x),color ='r',marker ='.')
plt.plot(x,y_mdp,color ='g',marker ='+')
plt.plot(x,y_rk4, color = 'm', marker = 'v')
plt.xlim([min(x),max(x)])
plt.ylim([min(y_fwd),100])
plt.grid()
plt.show()
    

def derivative(f,x,dx):
    y = np.zeros_like(x)
    for i in range(len(x)):
        if i==0:
            y[i] = (f(x[i+1]) - f(x[i])) / dx
        if i==len(x)-1:
            y[i] = (f(x[i]) - f(x[i-1])) / dx 
        else:
            y[i] = (f(x[i+1]) - f(x[i-1])) / (2 * dx)
    return y 

def second_derivative(f, x, dx):
    y = np.zeros_like(x)
    
    for i in range(len(x)):
        if i == 0:
            y[i] = (f(x[i+2]) - 2 * f(x[i+1]) + f(x[i])) / (dx * dx)
        elif i == len(x)-1:
            y[i] = (f(x[i]) - 2 * f(x[i-1]) + f(x[i-2])) / (dx * dx)
        else:
            y[i] = (f(x[i+1]) + f(x[i-1]) - 2 * f(x[i])) / (dx * dx)
    return y

def integral (f,x,dx,c=0):
    y = np.zeros_like(x)
    s = c
    for i in range(1,len(x)):
        s += (f(x[i-1]) + f(x[i])) * (dx / 2)
        y[i] = s
    return y

dx = 0.1
x = np.arange(0,10,dx)

def func_y(x):
    return x**2

dy = derivative(func_y,x,dx)
sy = integral(func_y,x,dx)

plt.plot(x,2*x,color ='b',marker ='.')
plt.plot(x,dy,color ='r',marker ='.')
plt.plot(x,sy,color ='g',marker ='.')
plt.xlim([min(x),max(x)])
plt.ylim([0,max(sy)])
plt.grid()
plt.show()


# SHO using rk4

def a_k(m,k,x):
    return - k * x**2 / m

m = 1
k = 0.5
x_0 = 1
v_0 = 0
a_0 = a_k(m,k,x_0)

dt = 0.1
t0, tf = 0, 10
t = np.arange(t0,tf,dt)

def rk4_xva(f_a,t,dt,x_0,v_0):
    a = np.zeros_like(t)
    v = np.zeros_like(t)
    x = np.zeros_like(t)

