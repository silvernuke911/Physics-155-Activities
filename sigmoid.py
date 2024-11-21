import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

telfile = 'telescope_general_data.csv'
df = pd.read_csv(telfile)
print(df)
print(df.info()) 
# df.fillna('0:00',inplace = True)
print(df)

def ms_ss_converter(value, mode):
    if pd.isnull(value):  # If the value is NaN, return it unchanged
        return value
    if mode == 'm2s':
        if isinstance(value,int):
            value = str(value)
        split_val = value.split(':')
        m = int(split_val[0])
        s = int(split_val[1])
        ss = m*60 + s 
        return ss
    elif mode == 's2m':
        m = value // 60
        s = value % 60
        ms = f'{m}:{s}'
        return ms
    else:
        raise ValueError('Wrong mode format')

time_lists = ['Assembly Time', 'Balancing Time', 'Alignment Time', 'Disassembly Time']
for header in time_lists:
    df[header] = df[header].apply(lambda x: ms_ss_converter(x, 'm2s') if pd.notnull(x) else x)

print(df)
df2023A = df[df['Batch'] == '2023A']
df2023B = df[df['Batch'] == '2023B']
df2024A = df[df['Batch'] == '2024A']

for batchdf in [df2023A , df2023B, df2024A ]:
    print(batchdf)


# Dictionary to store data for each batch
batches = {
    '2023A': df2023A,
    '2023B': df2023B,
    '2024A': df2024A
}

# Initialize containers for means and standard deviations
mean_scores = []
std_scores = []
years = []

# Loop through batches to compute mean and standard deviation
for year, df_batch in batches.items():
    mean_scores.append(df_batch['Score'].mean())
    std_scores.append(df_batch['Score'].std())
    years.append(float(year[:4]) + (0.5 if 'B' in year else 0))

# Plotting scores
plt.scatter(years, mean_scores)
plt.errorbar(years, mean_scores, yerr=std_scores, color='k', fmt='o',capsize = 2)
plt.xticks(years, batches.keys())
plt.show()

# Handling time-based calculations
mean_times = {batch: [df[xi].mean() for xi in time_lists] for batch, df in batches.items()}
std_times = {batch: [df[xi].std() for xi in time_lists] for batch, df in batches.items()}

# Example: Accessing mean and std of times for a specific batch
for batch in batches.keys():
    print(f"Mean times for {batch}: {mean_times[batch]}")
    print(f"Standard deviations for {batch}: {std_times[batch]}")

plt.scatter([0,1,2,3],mean_times['2023B'])
plt.errorbar([0,1,2,3],mean_times['2023B'],yerr = std_times['2023B'], capsize = 2)
plt.scatter([0,1,2,3],mean_times['2024A'])
plt.errorbar([0,1,2,3],mean_times['2024A'],yerr = std_times['2024A'], capsize = 2)
plt.ylim(0,1500)
plt.show()

