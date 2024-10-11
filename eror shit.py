import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Generate the original signal with noise
dx = 0.05
x = np.arange(-2*np.pi, 2*np.pi, dx)
smol = 0
a, b, n = -smol, smol, len(x)
cutoff = 20
y = 0.5 * x**2
y = 2*np.sin(x) + 0.1*np.cos(5*x)
y = np.sign(np.sin(x))
y_er = y + np.random.uniform(a, b, n)

# Plot the original signal
plt.plot(x, y, label='Original Signal')
plt.scatter(x, y_er, marker='.', color='k', label='Noisy Signal')
plt.grid()
plt.legend()
plt.show()

# FFT parameters
n, T = len(x), 1/len(x)
yf = np.fft.fft(y_er)           # Compute the FFT
xf = np.fft.fftfreq(n, T)[:n//2]  # Frequencies corresponding to the FFT

# Plot the original signal in time domain
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x, y_er)
plt.title('Noisy Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the FFT result (only the positive frequencies)
plt.subplot(1, 2, 2)
plt.plot(xf, 2.0/n * np.abs(yf[:n//2]))  # Magnitude of the FFT
plt.title('FFT of Signal (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.grid()
plt.show()

# Apply a Gaussian low-pass filter
cutoff_freq = cutoff  # Cutoff frequency in Hz
sigma = cutoff_freq / np.sqrt(2)  # Standard deviation for Gaussian filter
freqs = np.fft.fftfreq(n, T)      # Frequency components
gaussian_filter = np.exp(-0.5 * (freqs / sigma)**2)  # Gaussian filter

# Apply the Gaussian filter in frequency domain
filtered_yf_gaussian = yf * gaussian_filter

# Perform inverse FFT to get the filtered signal back in time domain
filtered_y_gaussian = np.fft.ifft(filtered_yf_gaussian)

# Plot the Gaussian filtered signal
plt.figure(figsize=(12, 6))

# Plot original noisy signal
plt.subplot(1, 2, 1)
plt.plot(x, y_er, label='Original Signal')
plt.title('Original Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

# Plot Gaussian filtered signal
plt.subplot(1, 2, 2)
plt.plot(x, filtered_y_gaussian.real, label='Gaussian Filtered Signal (Low-pass)', color='r')
plt.title('Gaussian Filtered Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Plot Gaussian-filtered FFT result
plt.figure(figsize=(6, 6))
plt.plot(xf, 2.0/n * np.abs(filtered_yf_gaussian[:n//2]), label='Gaussian Filtered FFT (Low-pass)')
plt.title('Gaussian Filtered FFT (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 20)  # Show frequencies from 0 to 20 Hz
plt.legend()
plt.grid()
plt.show()

### Butterworth Low-pass Filter ###
# Create a Butterworth filter with the same cutoff frequency
b, a = butter(N=4, Wn=cutoff_freq/(n/2), btype='low')  # N is the order, Wn is the normalized cutoff frequency

# Apply Butterworth filter to the noisy signal
filtered_y_butter = filtfilt(b, a, y_er)  # filtfilt applies the filter forward and backward for zero phase distortion

# Plot the Butterworth filtered signal
plt.figure(figsize=(12, 6))

# Plot original noisy signal
plt.subplot(1, 2, 1)
plt.plot(x, y_er, label='Original Signal')
plt.title('Original Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

# Plot Butterworth filtered signal
plt.subplot(1, 2, 2)
plt.plot(x, filtered_y_butter, label='Butterworth Filtered Signal (Low-pass)', color='b')
plt.title('Butterworth Filtered Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Plot FFT result for the Butterworth filtered signal
yf_butter = np.fft.fft(filtered_y_butter)

plt.figure(figsize=(6, 6))
plt.plot(xf, 2.0/n * np.abs(yf_butter[:n//2]), label='Butterworth Filtered FFT (Low-pass)')
plt.title('Butterworth Filtered FFT (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 20)  # Show frequencies from 0 to 20 Hz
plt.legend()
plt.grid()
plt.show()
