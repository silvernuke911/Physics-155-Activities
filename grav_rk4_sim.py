import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
from matplotlib.animation import PillowWriter
import matplotlib.font_manager as font_manager

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
        elapsed_milliseconds = int((elapsed_time -int(elapsed_time)) * 1_000)
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))
        remaining_str = f"{remaining_str}.{remaining_milliseconds:03d}"
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        elapsed_str = f"{elapsed_str}.{elapsed_milliseconds:03d}"
    else:
        elapsed_str   = '...'
        remaining_str = '...'
    print(f'|{bar}| {percent:.2f}% Time remaining: {remaining_str}  Time Elapsed = {elapsed_str}', end='\r')    # Print the progress bar with the remaining time

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
            new_filename = input('Input the new name of the file: ').strip()
            # If the user doesn't specify an extension, add the original extension
            if not new_filename.endswith(extension):
                new_filename += extension
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

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'axes.formatter.use_mathtext': True,
    'font.size': 12
})
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')

def compute_orbital_elements(r_i, v_i, G, M):
    if len(r_i)==2:                         # Creating 3d vectors if vectors are 2d
        r_i=np.append(r_i,0)
        v_i=np.append(v_i,0)
    mu = G * M                              # Gravitational parameter (mu = GM)
    h = np.cross(r_i, v_i)                  # Specific angular momentum vector (h = r x v)
    h_mag = np.linalg.norm(h)               # Magnitude of specific angular momentum
    e = (np.cross(v_i, h) / mu) - (r_i / np.linalg.norm(r_i))   # Eccentricity vector (e = ((v x h) / mu) - (r / |r|))
    e_mag = np.linalg.norm(e)               # Magnitude of eccentricity vector
    r_mag = np.linalg.norm(r_i)             # Semi-major axis (a)
    v_mag = np.linalg.norm(v_i)
    a = 1 / ((2 / r_mag) - (v_mag ** 2 / mu))
    T_p = 2 * np.pi * np.sqrt( a**3 / mu)   # Orbital period
    i = np.arccos(h[2] / h_mag)             # Inclination (i)
    k = np.array([0, 0, 1])                 # Node vector (n = k x h)
    n = np.cross(k, h)
    n_mag = np.linalg.norm(n)               # Magnitude of node vector
    if n_mag != 0:                          # Right ascension of the ascending node (RAAN, Ω)
        Omega = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0
    if n_mag != 0 and e_mag != 0:           # Argument of periapsis (ω)
        omega = np.arccos(np.dot(n, e) / (n_mag * e_mag))
        if e[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0
    if e_mag != 0:                          # True anomaly (ν)
        nu = np.arccos(np.dot(e, r_i) / (e_mag * r_mag))
        if np.dot(r_i, v_i) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0
    orbital_elements = {                    # Return orbital elements as a dictionary
        'semi_major_axis': a,
        'eccentricity': e_mag,
        'inclination': np.degrees(i),
        'LAN': np.degrees(Omega),
        'argument_of_periapsis': np.degrees(omega),
        'true_anomaly': np.degrees(nu),
        'orbital period': T_p
    }
    return orbital_elements

def kinematic_rk4(t, f, r0, v0):
    # Simulation loop
    time_start = time.time()
    dt = t[1] - t[0]  # Assuming uniform time steps
    half_dt = dt / 2
    # Initialize arrays to store positions, velocities, and accelerations
    dim = len(r0)
    r = np.zeros((len(t), dim))  # Position array
    v = np.zeros((len(t), dim))  # Velocity array
    a = np.zeros((len(t), dim))  # Acceleration array
    # Set initial conditions
    r[0] = r0
    v[0] = v0
    a[0] = f(t[0], r0, v0)
    for i in range(len(t) - 1):
        t_i = t[i]
        r_i = r[i]
        v_i = v[i]
        # RK4 coefficients for velocity
        k1_v = f(t_i, r_i, v_i)
        k2_v = f(t_i + half_dt, r_i + v_i * half_dt, v_i + k1_v * half_dt)
        k3_v = f(t_i + half_dt, r_i + v_i * half_dt, v_i + k2_v * half_dt)
        k4_v = f(t_i + dt, r_i + v_i * dt, v_i + k3_v * dt)
        # RK4 coefficients for position
        k1_r = v_i
        k2_r = v_i + k1_v * half_dt
        k3_r = v_i + k2_v * half_dt
        k4_r = v_i + k3_v * dt
        # Update velocity and position
        v[i + 1] = v_i + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * dt / 6
        r[i + 1] = r_i + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) * dt / 6
        # Update acceleration for the next step
        a[i + 1] = f(t[i + 1], r[i + 1], v[i + 1])
        progress_bar(i,len(t)-1,time_start)
    return r, v, a

# setting initial positions
G = 1
m = 1
r_i = np.array([1,0])
v_i = np.array([0,1.2])

def grav_acc(t,r,v, m=m):
    G = 1
    mu = G * m
    r_mag = np.linalg.norm(r)
    r_norm = lambda r, r_mag: r / r_mag if r_mag != 0 else r_mag*0
    return -(mu/r_mag**2) * r_norm(r, r_mag) #+ 0.01*v

orb_data = compute_orbital_elements(r_i,v_i,G,m)
period = orb_data['orbital period']
year = 2*np.pi
for element,value in orb_data.items():
    print(f' {element:^25} : {value:.3f}')

dt = 0.001

# t = np.arange(0,200+dt,dt)

t = np.linspace(0,2*period,5000)

r_rk4, v_rk4, a_rk4 = kinematic_rk4(t,grav_acc,r_i,v_i)

x_coord, y_coord = r_rk4[:,0], r_rk4[:,1]
radius = np.sqrt(x_coord**2 + y_coord**2)
theta = np.arctan(y_coord/x_coord)

def plot_r_theta():
    print('\n Plotting r(theta)')
    plt.plot(theta,radius, color = 'r',zorder = 2)
    plt.xlabel(r'$\theta$')
    plt.ylabel('$r$')
    plt.ylim([0,3])
    plt.grid()
    plt.show()

def plot_theta_t():
    print('\n Plotting theta(t)')
    plt.plot(t,theta, color = 'r',zorder = 2)
    plt.xlabel('$t$')
    plt.ylabel(r'$\theta$')
    plt.ylim([-np.pi,np.pi])
    plt.grid()
    plt.show()

def plot_orbit():
    print('\n Plotting Orbit')

    plt.plot(r_rk4[:,0],r_rk4[:,1],marker='',zorder=1,color='blue')
    # plt.plot(v[:,0],v[:,1],marker='',zorder=1,color='cyan')
    # plt.plot(a[:,0],a[:,1],marker='',zorder=1,color='magenta')

    plt.scatter(r_i[0],r_i[1],marker='.',s=60,color='red',zorder=2)
    plt.scatter(0,0,marker='o',s=80,color='yellow',zorder=3)

    plt.xlabel('$x$-axis (AU)')
    plt.ylabel('$y$-axis (AU)')
    plt.title(f'Planetary orbit simulation')

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_axisbelow(True)

    plt.grid()
    plt.show()
    print('Plotted')
# plot_orbit()
plot_r_theta()
plot_theta_t()
