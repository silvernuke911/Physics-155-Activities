import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

rng = np.random.default_rng()

def latex_font(): # Aesthetic choice
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 12
    })
latex_font()

########################
shape = 0.954  # Shape parameter (sigma)
scale = np.exp(4.5)  # Scale parameter (exp(mu))

# Generate random samples

# generate histogram
binchoice = np.logspace(0,3,100)

np.random.seed(42)
samples = stats.lognorm.rvs(shape, scale=scale, size=10000, loc = 0)
true_plot = stats.lognorm.pdf(binchoice, s = shape, loc=0, scale=scale)
bins, edges = np.histogram(samples, bins=binchoice, density=True)


def norm_pdf(x, s, loc, scale):
    return stats.lognorm.pdf(x, s=s, loc=loc, scale=scale)
opts, covs = optimize.curve_fit(lambda x, s: norm_pdf(x, s, 0, scale), edges[:-1], bins, p0=[shape])

##########################

# MLE shit
# Log-likelihood function for MLE
def log_likelihood(params, data):
    s, loc, scale = params
    return -np.sum(np.log(stats.lognorm.pdf(data, s=s, loc=loc, scale=scale)))

# Initial parameter guesses for MLE
initial_params = [shape, 0, scale]

# Optimize log-likelihood to get best-fit parameters
result = optimize.minimize(log_likelihood, initial_params, args=(samples,))
mle_shape, mle_loc, mle_scale = result.x

# Calculate the fitted PDF using MLE parameters
mle_pdf = stats.lognorm.pdf(binchoice, s=mle_shape, loc=mle_loc, scale=mle_scale)

########### Plotters
# Plotters

# Semilog
plt.scatter(edges[:-1], bins, label="Data", zorder = 2, color=(1.0, 0.2, 0.2))
plt.hist(samples, bins=binchoice, density = True, alpha = 0.35, zorder = 2, color = 'r')
plt.plot(binchoice, true_plot, "k--", label="True PDF")
plt.plot(binchoice, norm_pdf(binchoice, *opts, loc=0, scale=scale), "b-.", label="Fit from hist")
plt.plot(binchoice, mle_pdf,color = 'r', label="MLE Fit")
plt.semilogx()
plt.xlabel('$x$-axis (log scale)')
plt.ylabel('$y$-axis')
plt.title(r'\textbf{Lognormal distribution (semilog-$x$)}')
plt.grid(axis='x', which='both')
plt.grid(axis='y')
plt.legend()
plt.show()

# Linspace
plt.scatter(edges[:-1], bins, label="Data", zorder = 2, color=(1.0, 0.2, 0.2))
plt.hist(samples, bins=binchoice, density = True, alpha = 0.35, zorder = 2, color = 'r')
plt.plot(binchoice, true_plot, "k--", label="True PDF")
plt.plot(binchoice, norm_pdf(binchoice, *opts, loc=0, scale=scale), "b-.", label="Fit from hist")
plt.plot(binchoice, mle_pdf, color = "r", label="MLE Fit")
plt.grid()
plt.title(r'\textbf{Lognormal distribution (Linear spacing)}')
plt.xlabel('$x$-axis')
plt.ylabel('$y$-axis')
plt.legend()
plt.show()