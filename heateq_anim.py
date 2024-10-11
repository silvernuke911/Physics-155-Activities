import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
from matplotlib.animation import PillowWriter
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.formatter.use_mathtext']=True
mpl.rcParams.update({'font.size': 12})

def progress_bar(progress, total, start_time, scale=0.50):
    # Creates a progress bar on the command line, input is progress, total, and a present start time
    # progress and total can be any number, and this can be placed in a for or with loop

    percent = 100 * (float(progress) / float(total))                        # Calculate the percentage of progress
    bar = '█' * round(percent*scale) + '-' * round((100-percent)*scale)     # Create the progress bar string
    elapsed_time = time.time() - start_time                                 # Calculate elapsed time
    if progress > 0:                                                        # Estimate total time and remaining time
        estimated_total_time = elapsed_time * total / progress
        remaining_time = estimated_total_time - elapsed_time
        remaining_seconds = int(remaining_time)
        remaining_milliseconds = int((remaining_time - remaining_seconds) * 1_000)
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))
        remaining_str = f"{remaining_str}.{remaining_milliseconds:03d}"
    else:
        remaining_str = '...'
    print(f'|{bar}| {percent:.2f}% Time remaining: {remaining_str}  ', end='\r')    # Print the progress bar with the remaining time
    if progress == total: 
        elapsed_seconds = int(elapsed_time)
        elapsed_ms=int((elapsed_time-elapsed_seconds)*1000)                         # Print elapsed time when complete
        elapsed_seconds =  time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print('\n'+f'Elapsed time : {elapsed_seconds}.{elapsed_ms:03d}')

def rk4_heat_equation_neumann(T0, L, alpha, dt, dx, t_end):
    """
    Solve the heat equation using RK4 method with Neumann boundary conditions (dT/dx = 0 at boundaries).

    Parameters:
    T0 : numpy.ndarray
        Initial temperature distribution (array of length N).
    L : float
        Length of the rod.
    alpha : float
        Thermal diffusivity.
    dt : float
        Time step size.
    dx : float
        Spatial step size.
    t_end : float
        End time.

    Returns:
    t_values : numpy.ndarray
        The array of time values.
    T_values : numpy.ndarray
        The array of temperature distributions at each time step.
    """
    N = len(T0)        # Number of spatial points
    M = int(t_end / dt)  # Number of time steps

    # Initialize arrays
    t_values = np.linspace(0, t_end, M + 1)
    T_values = np.zeros((M + 1, N))
    T_values[0, :] = T0

    # Define the spatial derivative matrix with Neumann boundary conditions
    def spatial_derivative(T):
        D = np.zeros((N, N))
        for i in range(1, N - 1):
            D[i, i - 1] = 1
            D[i, i] = -2
            D[i, i + 1] = 1
        D = D / (dx ** 2)
        
        # Adjust for Neumann boundary conditions
        D[0, 0] = -2 / (dx ** 2)  # Boundary condition at x=0
        D[0, 1] = 2 / (dx ** 2)
        D[-1, -2] = 2 / (dx ** 2)  # Boundary condition at x=L
        D[-1, -1] = -2 / (dx ** 2)
        
        return D

    # Define the RK4 system
    def rk4_step(T):
        D = spatial_derivative(T)
        k1 = dt * alpha * np.dot(D, T)
        k2 = dt * alpha * np.dot(D, T + 0.5 * k1)
        k3 = dt * alpha * np.dot(D, T + 0.5 * k2)
        k4 = dt * alpha * np.dot(D, T + k3)
        return T + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Perform the RK4 integration
    for i in range(M):
        T_values[i + 1, :] = rk4_step(T_values[i, :])
    
    return t_values, T_values

def check_and_prompt_overwrite(filename):
    extension = os.path.splitext(filename)[1]
    def get_user_input(prompt, valid_responses, invalid_input_limit=3):
        attempts = 0
        while attempts < invalid_input_limit:
            response = input(prompt).lower().strip()
            if response in valid_responses:
                return response
            print("Invalid input. Valid inputs are [ Y , N , YES , NO ]")
            attempts += 1
        print("Exceeded maximum invalid input limit. Operation aborted.")
        return 'ABORT'

    def handle_file_exists(filename):
        while True:
            response = get_user_input(f"{filename} already exists, do you want to overwrite it? (Y/N): ", ['yes', 'y', 'no', 'n'], 5)
            if response in ['yes', 'y']:
                print                       ('\nx---------------------WARNING---------------------x')
                sure_response = get_user_input("Are you really sure you want to OVERWRITE it? (Y/N): ", ['yes', 'y', 'no', 'n'], 5)
                if sure_response in ['yes', 'y']:
                    print("Proceeding with overwrite...")
                    return True, filename
                elif sure_response in ['no', 'n']:
                    print('Operation aborted.')
                    return False, filename
                elif sure_response == 'ABORT':
                    return False, filename
            elif response in ['no', 'n']:
                return handle_rename(filename)
            elif response == 'ABORT':
                return False, filename

    def handle_rename(filename):
        while True:
            rename_response = get_user_input('Would you like to rename it? (Y/N): ', ['yes', 'y', 'no', 'n'],3)
            if rename_response in ['yes', 'y', '1']:
                return get_new_filename()
            elif rename_response in ['no', 'n', '0']:
                print('Operation aborted.')
                return False, filename
            elif rename_response == 'ABORT':
                return False, filename

    def get_new_filename():
        while True:
            new_filename = input('Input the new name of the file: ').strip() + extension
            if new_filename == ('ABORT' + extension):
                print('Operation aborted.')
                return False, new_filename
            if not os.path.isfile(new_filename):
                print(f'Proceeding with creation of {new_filename}')
                return True, new_filename
            print(f'{new_filename} already exists. Please put another file name.')

    if os.path.isfile(filename):
        return handle_file_exists(filename)
    return True, filename

# shape_functions
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
def dirac_delta_function(x,a):
    y=np.zeros_like(x)
    for i,el in enumerate(x):
        if np.isclose(el,a):
            y[i]=1
    return y
def square_func(x, a, b):
    # Initialize an array of zeros with the same shape as x
    result = np.zeros_like(x)
    
    # Set the elements to 1 where the condition a <= x <= b is met
    result[(x >= a) & (x <= b)] = 1
    
    return result
def sin_func(x,amp = 1,phase = 2*np.pi, alt = 0):
    result = amp* np.sin(x - phase) + alt
    return result

def gaussian_func(x,height=1,spread=1, offset=0, alt=0):
    return height * (1 / (spread * np.sqrt(2*np.pi)))*np.exp(-0.5 * ((x-offset)/spread)**2) + alt

bar_start, bar_length = 0, 5
dx = 0.075
dt = 0.05
x = np.arange(bar_start,bar_length + dx,dx)
t_0, t_f = 0, 20 + dt
alpha = 0.06
# T0 = gaussian_func(x, offset = bar_length/2, spread = 0.5)
# T0 = square_func(x, 1.25, 3.75)
# T0 = np.sin(5*x)
T0 = unit_step(x,2.5)
simulation = rk4_heat_equation_neumann(T0, bar_length, alpha, dt, dx, t_f)
filename = "heat_eq_rk4_5.gif"

def plot():
    plt.plot(x, T0, color = 'red')
    plt.grid()
    plt.xlim([0,bar_length])
    plt.ylim([min(T0)-.25,max(T0)*(1.25)])
    plt.xlabel(r'$x$-axis')
    plt.ylabel(r'Temperature ($T$/$T_{\text{max}}$)')
    plt.title('1D Heat Equation Solver')
    plt.show()
    print('Plotted!')
plot()

def animated_plot(filename):
    metadata = dict(title='Movie', artist='silver')
    writer = PillowWriter(fps=20, metadata=metadata)
    fig = plt.figure()
    start_time=time.time()
    t,T = simulation
    max_time=np.max(simulation[0])
    name_doesnt_exists, filename = check_and_prompt_overwrite(filename)
    if not name_doesnt_exists:
        return
    print('Animating file ...')
    with writer.saving(fig, filename,100):
        for i in range(len(T)):
            plt.plot(x,T[i],color='red',marker='')
            plt.xlabel(r'$x$-axis')
            plt.ylabel(r'Temperature ($T$/$T_{\text{max}}$)')
            plt.xlim([bar_start, bar_length])
            plt.ylim([min(T0)-.25,max(T0)*(1.25)])
            plt.title(f'1D Heat Equation Solver : t={t[i]:.2f}')
            
            plt.grid()
            writer.grab_frame()
            # plt.pause(2)
            plt.pause(0.001)
            plt.clf()
            progress_bar(t[i],max_time,start_time)
        print("\n"+"File animated and saved!")

animated_plot(filename)

