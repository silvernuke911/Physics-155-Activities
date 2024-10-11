import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def numerical_differentiator(x, func):
    """
    Returns the array of the numerical derivative of a function f(x), dy/dx.
    
    Parameters:
        x (np.ndarray): Array of numbers representing values on the x axis.
        func (callable): A function taking x as an input.
    
    Returns:
        np.ndarray: Array of numerical derivative values.
    """
    y = func(x)  # Creating the function values
    output = np.zeros_like(y)  # Initializing an array
    
    # Calculate differences for the interior points using central difference
    dx = np.diff(x)
    dy = np.diff(y)
    output[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    
    # Forward difference for the first point
    output[0] = (y[1] - y[0]) / dx[0]
    
    # Backward difference for the last point
    output[-1] = (y[-1] - y[-2]) / dx[-1]
    
    return output

def numerical_integrator(x, func, c=0):
    """
    Returns the array of the numerical integral of a function y(x), ∫ y(x) dx.

    Parameters:
        x (np.ndarray): Array of numbers representing values on the x axis.
        func (callable): A function taking x as an input.
        c (float): A constant of integration, default is 0.

    Returns:
        np.ndarray: Array of numerical integral values.
    """
    y = func(x)  # Creating the function values
    output = np.zeros_like(y)  # Initializing an array

    # Using the trapezoidal rule for integration
    dx = np.diff(x)
    mid_y = (y[:-1] + y[1:]) / 2  # Midpoints of y values for trapezoidal rule

    # Calculate cumulative sum of areas of trapezoids
    cumulative_sum = np.cumsum(mid_y * dx)
    
    output[1:] = cumulative_sum
    output[0] = c  # Initial value with the constant of integration
    
    output += c  # Add constant of integration to all values
    
    return output

def numerical_partial_derivative_2d(x,y,func):
    # Returns the array of the partial derivative of a 2 dimensional function f(x,y) with respect to x, df(x,y)/dx
    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   y       : a constant value of the variable y
    #                   func    : a function taking x and y as an input

    z=func(x,y)                                     # Creating the function values
    output=np.zeros_like(z)                         # Initializing an array                              
    for i in range(len(x)):                         # Differentiation loop
        if i-1==-1:
            m=(z[i+1]-z[i])/(x[i+1]-x[i])
        elif i+1==len(x):
            m=(z[i]-z[i-1])/(x[i]-x[i-1])
        else:
            m=(z[i+1]-z[i-1])/(x[i+1]-x[i-1])
        output[i]=m
    return output

def numerical_differentiator_array(x, y):
    """
    Returns the array of the numerical derivative of an array of function values y and independent variable x.

    Parameters:
        x (np.ndarray): Array of numbers representing values on the x axis.
        y (np.ndarray): Array of numbers representing function values dependent on x.

    Returns:
        np.ndarray: Array of numerical derivative values.
    """
    if len(x) != len(y):  # Check array length compatibility
        raise ValueError("Lengths of x and y must match")
    
    output = np.zeros_like(y)  # Output array initialization
    
    # Using central differences for the interior points
    output[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    
    # Using forward difference for the first point
    output[0] = (y[1] - y[0]) / (x[1] - x[0])
    
    # Using backward difference for the last point
    output[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    
    return output

def numerical_integrator_array(x, y, c=0):
    """
    Returns the array of the numerical integral of an array of function values y and independent variable x.

    Parameters:
        x (np.ndarray): Array of numbers representing values on the x axis.
        y (np.ndarray): Array of numbers representing function values dependent on x.
        c (float): Constant of integration (default is 0).

    Returns:
        np.ndarray: Array of numerical integral values.
    """
    if len(x) != len(y):  # Check array length compatibility
        raise ValueError("Lengths of x and y must match")
    
    output = np.zeros_like(y)  # Output array initialization
    s = 0  # Sum initialization
    
    for i in range(len(x)):  # Integration loop
        if i == 0:
            s += c
        else:
            s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])  # Trapezoidal rule
        output[i] = s
        
    return output

def numerical_second_derivative(x, y):
    """
    Returns the array of the second derivative of an array of function values y and independent variable x

    Parameters:
    x (np.ndarray): An array of numbers representing values on the x-axis
    y (np.ndarray): An array of numbers of the function value dependent on x

    Returns:
    np.ndarray: An array of second derivative values
    """
    # Initialize the output array
    second_deriv = np.zeros_like(y)
    
    # Compute the second derivative using central difference method
    for i in range(1, len(x) - 1):
        second_deriv[i] = (y[i+1] - 2*y[i] + y[i-1]) / (0.5*(x[i+1] - x[i-1]))**2

    # Handle boundaries with forward and backward difference
    second_deriv[0] = (y[2] - 2*y[1] + y[0]) / (x[1] - x[0])**2
    second_deriv[-1] = (y[-1] - 2*y[-2] + y[-3]) / (x[-1] - x[-2])**2

    return second_deriv

def numerical_second_derivative2(x,y):
    
    # Returns the array of the second derivative of an array of function values y and independent variable x

    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   y       : an array of numbers of the function value dependent on x

    first_deriv=numerical_differentiator_array(x,y)     # Calling first function derivative
    second_deriv=numerical_differentiator_array(x,first_deriv)  # Calling second function derivative
    return second_deriv

def numerical_second_integral(x,y,c1=0,c2=0):
        
    # Returns the array of the second integral of an array of function values y and independent variable x

    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   y       : an array of numbers of the function value dependent on x
    #                   c1      : constant of integration for the first integral, default is 0
    #                   c2      : constant of integration for the second integral, default is 0

    first_integral=numerical_integrator_array(x,y,c1)             # Calling first function integral
    second_integral=numerical_integrator_array(first_integral,c2) # Calling second function integral
    return second_integral

t_prev=0
def numerical_heat_function(x,t,t_prev,func,alpha=1):
    # Inputs        x       : An array of x positions
    #               func    : A function describing the temperature at position x
    #               t       : current time value
    #               t_prev  : time value from one iteration ago
    #               alpha   : thermal diffusivity constant, default is 1

    T = func(x)             # Temperature distribution function

    def mean(*nums):
        return sum(nums)/len(nums)
    def neighbor_ave_temp(x,T):     # Calculate the average temperature of the neigboring points of a point x
        neigbor_ave=np.array([])
        for i in range(len(x)):
            temp=0
            if i==0:
                temp=mean(T,T[i+1])
            if i==(len(x)-1):
                temp=mean(T[i-1],T)
            else:
                temp=mean(T[i-1],T[i+1])
            neigbor_ave=np.append(neigbor_ave,temp)
        return neigbor_ave
    
    def temp_update():
        neighbor_temp=neighbor_ave_temp(x,T)
        for i in range(len(x)):
            T[i]=T[i]+alpha*(neighbor_temp[i]-T[i])*(t-t_prev)

    T=temp_update()
    return T
            
def unit_step(x, a):
    """
    Returns a Heaviside unit step function array from 0 to 1 at the boundary value a.

    Parameters:
        x (np.ndarray): Array of x values.
        a (float): Boundary value.

    Returns:
        np.ndarray: Heaviside unit step function array.
    """
    output = np.zeros_like(x)
    output[x >= a] = 1
    return output

def connecting_function(arr, x1, x2):
    """
    Create a function that smoothly transitions from y=0 to y=1 
    on the interval [x1, x2].
    
    Parameters:
        arr (np.ndarray): Array of values where the function will be evaluated.
        x1 (float): Lower boundary of the transition interval.
        x2 (float): Upper boundary of the transition interval.
    
    Returns:
        np.ndarray: Array of values representing the smoothly transitioning function.
    """
    # Inner function psi(x): Monotone increasing function from [0, 1]
    def psi(x):
        return x ** 3
    # Inner function phi(x): Smoothly interpolates between 0 and 1 in the interval [x1, x2]
    def phi(x):
        alph = (x - x1) / (x2 - x1)
        return psi(alph) / (psi(alph) + psi(1 - alph))
    # Initialize the output array with zeros
    out = np.zeros_like(arr)
    # Indices where x is between x1 and x2
    mask = (arr >= x1) & (arr < x2)
    # Calculate values using phi function for indices in the mask
    out[mask] = phi(arr[mask])
    # Set values to 1 where x >= x2
    out[arr >= x2] = 1
    
    return out

def double_slit_experiment(x, slit_distance=0.05, slit_width=0.05, wavelength=0.0000650, screen_distance=100):
    """
    Returns an intensity function of the double slit experiment.

    Parameters:
    x (np.ndarray): Positions on the screen.
    slit_distance (float): Distance between the two slits (default 0.05).
    slit_width (float): Width of each slit (default 0.05).
    wavelength (float): Wavelength of the light used (default 0.0000650).
    screen_distance (float): Distance between the slits and the screen (default 100).

    Returns:
    np.ndarray: Intensity pattern on the screen.
    """
    theta = x / screen_distance
    pi_over_wavelength = np.pi / wavelength
    
    # Vectorized sinc term calculation
    sinc_term = np.sinc(pi_over_wavelength * slit_width * np.sin(theta) / np.pi)
    
    # Vectorized cos term calculation
    cos_term = np.cos(pi_over_wavelength * slit_distance * np.sin(theta))
    
    # Calculate intensity pattern
    intensity = (cos_term ** 2) * (sinc_term ** 2)
    
    return intensity

def square_wave(x, frequency=1, amplitude=1, phase=0, offset=0):
    """
    Generates a square wave based on the input parameters.
    
    Parameters:
    x (np.ndarray): Array of time or independent variable values.
    frequency (float): Frequency of the square wave. Default is 1.
    amplitude (float): Amplitude of the square wave. Default is 1.
    phase (float): Phase shift of the wave. Default is 0.
    offset (float): Vertical offset of the wave. Default is 0.
    
    Returns:
    np.ndarray: Array of square wave values.
    """
    # Calculate the sine wave
    sine_wave = np.sin(2 * np.pi * frequency * x + phase)
    
    # Generate the square wave
    square_wave = amplitude * np.sign(sine_wave) + offset
    
    return square_wave

def sine_wave(x, frequency=1, amplitude=1, phase=0, offset=0):
    """
    Generates a sine wave based on the input parameters.
    
    Parameters:
    x (np.ndarray): Array of time or independent variable values.
    frequency (float): Frequency of the sine wave. Default is 1.
    amplitude (float): Amplitude of the sine wave. Default is 1.
    phase (float): Phase shift of the wave. Default is 0.
    offset (float): Vertical offset of the wave. Default is 0.
    
    Returns:
    np.ndarray: Array of sine wave values.
    """
    # Precompute constant to avoid repetition in the calculation
    angular_frequency = 2 * np.pi * frequency
    
    # Calculate the sine wave
    y = amplitude * np.sin(angular_frequency * x + phase) + offset
    
    return y

def triangle_wave(x, frequency=1, amplitude=1, phase=0, offset=0):
    """
    Generates a triangle wave based on the input parameters.
    
    Parameters:
    x (np.ndarray): Array of time or independent variable values.
    frequency (float): Frequency of the triangle wave. Default is 1.
    amplitude (float): Amplitude of the triangle wave. Default is 1.
    phase (float): Phase shift of the wave. Default is 0.
    offset (float): Vertical offset of the wave. Default is 0.
    
    Returns:
    np.ndarray: Array of triangle wave values.
    """
    # Compute the period of the wave
    period = 1 / frequency
    
    # Compute the wave's fractional part and adjust for phase shift
    fractional_part = np.mod((x - phase), period) / period
    
    # Compute the absolute distance from the midpoint and scale
    y = 4 * amplitude * np.abs(fractional_part - 0.5) - amplitude
    
    # Apply the vertical offset
    y += offset
    
    return y

def sawtooth_wave(x, frequency=1, amplitude=1, phase=0, offset=0):
    """
    Generates a sawtooth wave based on the input parameters.
    
    Parameters:
    x (np.ndarray): Array of time or independent variable values.
    frequency (float): Frequency of the sawtooth wave. Default is 1.
    amplitude (float): Amplitude of the sawtooth wave. Default is 1.
    phase (float): Phase shift of the wave. Default is 0.
    offset (float): Vertical offset of the wave. Default is 0.
    
    Returns:
    np.ndarray: Array of sawtooth wave values.
    """
    # Calculate the normalized phase
    normalized_phase = (x - phase) * frequency
    
    # Compute the sawtooth wave using the modulo operation
    y = amplitude * (normalized_phase - np.floor(normalized_phase))
    
    # Apply the vertical offset
    y += offset
    
    return y

def RC_circuit(t, v, R, C):
    """
    Simulate the transient response of an RC circuit to an input signal.
    
    Parameters:
        t (np.ndarray): Array of time points.
        v (np.ndarray): Input signal (e.g., voltage).
        R (float): Resistance value of the resistor (ohms).
        C (float): Capacitance value of the capacitor (farads).
    
    Returns:
        np.ndarray: Simulated output signal of the RC circuit.
    """
    # Calculate time step
    dt = np.diff(t)
    dt = np.mean(dt) if len(dt) > 0 else 1.0
    
    # Calculate time constant
    tau = R * C
    
    # Initialize output array
    out = np.zeros_like(v)
    
    # Initialize previous time and previous output
    t_prev = t[0]
    y_res = 0
    
    # Iterate over time points
    for i in range(1, len(t)):
        t_curr = t[i]
        
        # Calculate response of RC circuit using trapezoidal rule
        response = (v[i] - y_res) * (t_curr - t_prev) / tau
        
        # Update previous time
        t_prev = t_curr
        
        # Update response
        y_res += response
        
        # Store response in output array
        out[i] = y_res
    
    return out

def PID_controller(x, y_desired, y_initial, K_p, K_i, K_d):
    """
    Simulate a Proportional-Integral-Derivative (PID) controller.

    Parameters:
        x (np.ndarray): Array of time points.
        y_desired (np.ndarray): Array of desired values for the controlled variable.
        y_initial (float): Initial value of the controlled variable.
        K_p (float): Proportional gain.
        K_i (float): Integral gain.
        K_d (float): Derivative gain.

    Returns:
        tuple: A tuple containing:
            - control_list (list): List of simulated values of the controlled variable.
            - error_list (list): List of errors between desired and simulated values.
    """
    # Initialize variables
    lag = 0.1
    y_response = y_initial
    error = 0
    error_prev = 0
    integral = 0
    t_prev = x[0]
    control_list = [y_initial]  # Initialize with initial value
    error_list = []

    def PID(y_desired, y_measured, K_p, K_i, K_d, dt):
        """
        Calculate PID response at each time step.

        Parameters:
            y_desired (float): Desired value of the controlled variable.
            y_measured (float): Current value of the controlled variable.
            K_p (float): Proportional gain.
            K_i (float): Integral gain.
            K_d (float): Derivative gain.
            dt (float): Time step.

        Returns:
            tuple: A tuple containing:
                - response (float): Control output at the current time step.
                - error (float): Error between desired and measured values.
        """
        nonlocal error, error_prev, integral

        # Calculate error
        error = y_desired - y_measured

        # Proportional term
        P = K_p * error

        # Integral term
        integral += K_i * error * dt
        I = integral

        # Derivative term
        D = K_d * (error - error_prev) / dt
        error_prev = error

        # Total response
        response = P + I + D

        return response, error

    # Iterate over time points
    for i in range(1, len(x)):
        # Calculate time step
        dt = x[i] - t_prev

        # Calculate PID response
        response, error = PID(y_desired[i], y_response, K_p, K_i, K_d, dt)

        # Update controlled variable
        y_response += response

        # Append values to lists
        control_list.append(y_response)
        error_list.append(error)

        # Update previous time
        t_prev = x[i]

    return control_list, error_list
x_min =  0
x_max =  1
y_min = -5
y_max =  5

# def func(x):
#     return (x*y)**2

def func(x):
    return unit_step(x,0.5)

x = np.arange(x_min,x_max,0.001)
y = np.arange(y_min,y_max,0.01)
z=func(x)

def zlims(z):
    yran=(np.max(z)-np.min(z))
    yave=(np.max(z)+np.min(z))/2
    return [yave-0.75*yran,yave+0.75*yran]
z_min = zlims(z)[0]
z_max = zlims(z)[1]

X, Y = np.meshgrid(x, y)
Z = func(X, Y)

t=np.arange(0,3,0.01)
def plot_2d():
    # dz = numerical_differentiator(x,func)
    # Fz = numerical_integrator(x,func)
    plt.axis([x_min,x_max, z_min, z_max])
    plt.grid()
    plt.plot(x,z,color='red')
    # plt.plot(x,dz,color='blue')
    # plt.plot(x,Fz,color='green')
    plt.show()

plot_2d()

def plot_2d_animated():
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    delta_time=0.01
    
    for t_val in t:
        z=numerical_heat_function(x,t,func,delta_time)
        name=f't value : {t_val:.2f}'
        plt.title(name)
        plt.axis([x_min,x_max, z_min, z_max])
        plt.grid()
        plt.plot(x,z,color='red')
        plt.pause(0.001)
        plt.clf()
    plt.close()

# plot_2d_animated()


def plot_3d():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)
    ax.set_zlim(z_min,z_max)
    surf=ax.plot_surface(X, Y, Z, cmap='inferno')
    fig.colorbar(surf,ax=ax,shrink=0.75,aspect=15,label='temperature',extend="max",location="right")
    plt.show()
# plot_3d()
