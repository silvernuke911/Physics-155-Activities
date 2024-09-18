import numpy as np
import matplotlib.pyplot as plt
import time

# True function
def sqwave_true(x, frequency=0.25, amplitude=1, phase=0, offset=0):
    y = amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset
    output = np.zeros_like(x)
    for i, y_val in enumerate(y):
        if y_val >= 0:
            output[i] = amplitude
        else:
            output[i] = -amplitude
    return output

# Approximate function
def sqwave_approx(x, niters):
    output = np.zeros_like(x)
    def sin_sum(x, max_iter):
        out = 0
        for i in range(max_iter):
            out += np.sin((2 * i + 1) * x) / (2 * i + 1)
        return out
    for i, x_val in enumerate(x):
        output[i] = (2 / np.pi) * sin_sum(x_val, niters)
    return output

# Timper function
def time_function(func, dtype):
    x = np.linspace(-4 * np.pi, 4 * np.pi, 1000, dtype=dtype)
    start_time = time.time()
    y = func(x)
    end_time = time.time()
    return end_time - start_time

# Data type list
dtype_list = [np.float16, np.float32, np.float64, np.longdouble]

# Time the functions for each dtype
for dtype_type in dtype_list:
    true_time = time_function(lambda x: sqwave_true(x, 0.5 / np.pi, 0.5, 0, 0), dtype_type)
    approx_time = time_function(lambda x: sqwave_approx(x, 150), dtype_type)
    print(f"Data Type: {dtype_type}, sqwave_true time: {true_time:.6f}s, sqwave_approx time: {approx_time:.6f}s")
