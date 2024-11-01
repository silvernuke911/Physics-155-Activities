import numpy as np
import matplotlib.pyplot as plt

# Define the function you want to approximate with a Fourier series
def func(x):
    # Example: a square wave function
    return np.sin(x)+0.5*np.cos(2*x+1)

def square_wav(x, A=1, T = (np.pi / 8)):
    return A * np.sign(np.sin(x/T))

def sawtooth(x, T=np.pi):
    return 2 * (x / T - np.floor(0.5 + x / T))

# Fourier series approximation function
def fourier_series(func, x, N, T):
    a0 = (2 / T) * np.trapz(func(x), x)  # Compute a0 using numerical integration (trapezoidal rule)
    approx = a0 / 2  # Start with a0/2
    
    for n in range(1, N + 1):
        cos_term = np.cos(2 * np.pi * n * x / T)
        sin_term = np.sin(2 * np.pi * n * x / T)
        
        # Compute an and bn coefficients using numerical integration
        an = (2 / T) * np.trapz(func(x) * cos_term, x)
        bn = (2 / T) * np.trapz(func(x) * sin_term, x)
        
        # Add the nth term to the approximation
        approx += an * cos_term + bn * sin_term
        
    return approx

# Parameters for the Fourier series
N_max = 10  # Maximum number of terms in the Fourier series
T = 2 * np.pi  # Period of the function
dx = 0.01
x = np.arange(0, 2 * np.pi, dx)  # Sampling points

# Compute the Fourier series for increasing number of terms
for N in [1, 3, 5, 10, 20, 50,100,200]:
    y_approx = fourier_series(func, x, N, T)
    
    # Plot the original function and the Fourier series approximation
    plt.figure(figsize=(8, 4))
    plt.plot(x, func(x), label='Original function', color='b')
    plt.plot(x, y_approx, label=f'Fourier Series Approximation (N={N})', color='r', linestyle='-')
    plt.title(f'Fourier Series Approximation with N={N}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


A = np.random.rand(10,10)
xlims = [-5,5]
ylims = [-5,5]
plt.imshow(A, extent = [*xlims, *ylims])