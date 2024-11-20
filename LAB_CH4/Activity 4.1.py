import matplotlib.pyplot as plt
import numpy as np

# Setting constants
R = 1
v_0 = 5

# Given Example Solution
A = np.array([[ 1,0,0],
              [-1,0,0],
              [-1,1,0],
              [0,-1,0],
              [0,-1,1],
              [0,0,-1]])
K = (1/R)*np.eye(6)
V = np.array([[v_0,0,0,0,0,0]]).T
x = np.linalg.solve(A.T@K@A,A.T@K@V)
print(x)

# Make a function that creates a matrix that solves that resistor matrix given n junctions

def resistor_solve(n, R = 1,V0 = 5):
    def generate_A(n):
        output = np.zeros((n, 2 * n))
        for i in range(n):
            if 2 * i < 2 * n:
                output[i, 2 * i] = 1  
            if 2 * i + 1 < 2 * n:
                output[i, 2 * i + 1] = -1  
            if 2 * i + 2 < 2 * n:
                output[i, 2 * i + 2] = -1 
        return output.T
    A = generate_A(n)
    K = (1/R) * np.eye(2*n)
    V = np.zeros(2*n)
    V[0]= V0
    V = np.array(V).T
    X = np.linalg.solve(A.T@K@A,A.T@K@V)
    return(X)

def latex_font(): 
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 12
    })
latex_font()

N = 30
index = range(N)
e_res = resistor_solve(N)
plt.grid()
plt.scatter(index, e_res, marker = '.', color = 'r', zorder = 2)
plt.xlabel('Junction index $e_i$', size = 15)
plt.ylabel('Voltage Drop (V)', size = 15)
plt.show()
