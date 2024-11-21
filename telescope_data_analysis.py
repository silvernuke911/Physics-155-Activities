import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def latex_font2(): 
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 12
    })
# latex_font2()

tel_data = 'telescope_data.csv'

# Load data using pandas for efficient handling
df = pd.read_csv(tel_data, delimiter='\t', header=None, names=['Assembly', 'Balancing', 'Alignment', 'Disassembly'])

print(df)

# Extract individual columns as NumPy arrays
assembly_times = df['Assembly'].values
balancing_times = df['Balancing'].values
alignment_times = df['Alignment'].values
disassembly_times = df['Disassembly'].values

# Define histogram parameters
dt = 120
bins = np.arange(0, 1500 + dt, dt)

# Plot histograms
plt.hist(assembly_times, bins, color='r', label='Assembly')
plt.hist(balancing_times, bins, color='b', alpha=0.75, label='Balancing')
plt.hist(alignment_times, bins, color='g', alpha=0.75, label='Alignment')
plt.hist(disassembly_times, bins, color='k', alpha=0.75, label='Disassembly')

# Format x-axis labels to show minutes:seconds
xticks = np.arange(0, 1500 + dt, 60)  # Every 2 bins
xtick_labels = [f"{x // 60}:{x % 60:02}" for x in xticks]
plt.xticks(xticks, xtick_labels, rotation=45)

# Add labels and legend
plt.xlim(0,1500)
plt.xlabel('Time (m:s)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()