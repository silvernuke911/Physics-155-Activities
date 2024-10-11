import numpy as np 
import matplotlib.pyplot as plt

low, high = 0, 10
dx = 0.01
x = np.arange(low,high,dx)
c = 0
def func(x):
    return x
y = func(x)
dy = np.gradient(func(x),x)
int_y = np.array([np.trapz(y[:i+1], x[:i+1]) for i in range(len(x))])

plt.plot(x,y, color = 'r')
plt.plot(x,dy, color = 'k')
plt.plot(x,int_y,color = 'b')
plt.grid()
plt.show()