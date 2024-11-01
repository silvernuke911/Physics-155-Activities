import numpy as np
import matplotlib.pyplot as plt

n = 6
numdice = 5
def dice(n):
    nums = range(1,n+1)
    return np.random.choice(nums)

bin_index = range(numdice, numdice * n + 1)
bins = np.zeros(len(bin_index))
num_rolls = 10000
for i in range(num_rolls):
    result_list = np.zeros(numdice)
    for j in range(numdice):
        result_list[j] = dice(n)
    c = int(np.sum(result_list))
    bins[c-numdice] += 1

print(list(bin_index))
print(bins)
plt.bar(bin_index,bins, color = 'r')
plt.xlim(bin_index[0]-.5, bin_index[-1]+.5)
plt.show()


