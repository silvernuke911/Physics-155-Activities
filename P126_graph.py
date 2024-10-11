import numpy as np
import matplotlib.pyplot as plt

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

def small_circle(center,radius):
    theta = np.linspace(0, 2*np.pi, 360)
    x = center[0] + 0.3 * radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x,y


def U_func(x, U_0, a):
    U = U_0 * (2 * (x/a)**2 - (x/a)**4)
    return U

lims = 2
x = np.linspace(-lims,lims,200)
U_0 = 1
a = 1

latex_font()
plt.xlim([-lims,lims])
plt.ylim([-8,2])
# plt.title('Potential Energy function for a particle')
plt.xlabel(r'Radius $r$ $(a)$')
plt.ylabel(r'Potential Energy $U$ $(U_0)$')
plt.plot(x,U_func(x,U_0,a), color = 'blue')
plt.plot(small_circle([1,1],0.2)[0],small_circle([1,1],0.2)[1], color = 'red')
plt.plot(small_circle([-1,1],0.2)[0],small_circle([-1,1],0.2)[1], color = 'red')
plt.plot(small_circle([0,0],0.2)[0],small_circle([0,0],0.2)[1], color = 'green')
ax = plt.gca()
ax.set_aspect(0.3, adjustable='box')
ax.set_axisbelow(True)
plt.grid(True)
plt.show()

def Lenard_Jones_U(r, r_0, epsilon):
    U = epsilon*((r/r_0)**12 - 2 * (r/r_0)**6)
    return U

x = np.linspace(-1.1,1.1,1000)
y = Lenard_Jones_U(x,1,1)

plt.plot(x,y,color='b')
plt.grid()
plt.show()

def generate_primes(n):
    if n < 1:
        return []
    primes = np.zeros(n, dtype=int)  # Pre-allocate an array for n primes
    primes[0] = 2  # The first prime is 2
    count = 1  # Number of primes found so far
    candidate = 3  # Start checking from 3 (since 2 is already known)
    while count < n:
        is_prime = True
        # Check divisibility only by known primes up to sqrt(candidate)
        limit = int(np.sqrt(candidate)) + 1
        for prime in primes[:count]:
            if prime > limit:  # No need to check beyond sqrt(candidate)
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes[count] = candidate
            count += 1
        candidate += 2  # Skip even numbers, only check odd candidates
    return primes
n=1000
prime_list = generate_primes(n)
print(f"The first {n} primes are: {prime_list}")

import decimal
from decimal import Decimal, getcontext

import decimal
from decimal import Decimal, getcontext

def generate_pi(n):
    """
    Generate pi to n decimal places using the Chudnovsky algorithm.
    
    Parameters:
    n (int): Number of decimal places of pi to generate.

    Returns:
    str: The value of pi up to n decimal places as a string.
    """
    getcontext().prec = n + 2  # Set precision to n decimal places + extra digits for internal rounding

    # Constants for Chudnovsky algorithm
    C = 426880 * Decimal(10005).sqrt()
    M = 1
    X = 1
    L = 13591409
    K = 6
    S = L

    for i in range(1, n * 2):  # Adjust iteration count for precision
        M = (M * (K**3 - 16*K)) // (i**3)
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / X
        K += 12

    pi = C / S
    return str(pi)[:n+2]  # Return pi up to n decimal places

# Example usage
n = 1000  # Number of decimal places
pi_value = generate_pi(n)
print(f"Pi to {n} decimal places is: {pi_value}")

