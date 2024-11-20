import matplotlib.pyplot as plt
import csv
import numpy as np

tel_data = 'telescope_data.csv'
tel_data_ave = 'telescope_data ave.csv'

assembly_times    = []
balancing_times   = []
alignment_times   = []
disassembly_times = []

with open(tel_data, mode = 'r') as file:
    reader = csv.reader(file, delimiter = '\t')
    for row in reader:
        assembly_time    = row[0]
        balancing_time   = row[1]
        alignment_time   = row[2]
        disassembly_time = row[3]
        assembly_times.append(int(assembly_time))
        balancing_times.append(int(balancing_time))
        alignment_times.append(int(alignment_time))
        disassembly_times.append(int(disassembly_time))
dt = 30
bins = np.arange(0, 1500+dt, dt)
plt.hist(assembly_times, bins, color ='r')
plt.hist(balancing_times, bins, color ='b', alpha = 0.75)
plt.hist(alignment_times, bins, color = 'g', alpha = 0.75)
plt.hist(disassembly_times, bins, color = 'k', alpha = 0.75)
plt.xlabel('Times')
plt.ylabel('Frequency')
plt.show()
