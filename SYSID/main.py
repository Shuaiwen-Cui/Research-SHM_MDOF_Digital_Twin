import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Import modules from the same directory
from newmark_beta import newmark_beta

# Define system parameters following the paper's methodology
# Number of degrees of freedom
nDOF = 5

# Time step - 100 Hz sampling rate
dt = 1.0 / 100.0  # seconds (100 Hz sampling rate)

# System parameters - 5-DOF shear building model
# Masses (kg) for each floor (from bottom to top)
m = np.array([7.5, 7.5, 7.5, 7.5, 7.5])

# Stiffnesses (N/m) for each floor - reduced by half
k = np.array([200000, 200000, 200000, 200000, 200000])  # 200 kN/m = 200000 N/m (half of original)

# Total number of time steps - 100 seconds at 100 Hz
nt = int(100.0 / dt)  # 100 seconds * 100 Hz = 10000 samples

# Non-proportional modal damping: assign mode-specific damping ratios
# Following paper: assign damping ratios to first nDOF vibration modes
zeta_modal = np.array([0.02, 0.015, 0.025, 0.02, 0.018])  # Mode-specific damping ratios

# Generate Gaussian white noise excitation
# Simple white noise without frequency filtering
np.random.seed(42)  # For reproducibility
noise_std = 100.0  # Standard deviation of white noise (N)
force = np.random.normal(0, noise_std, (nDOF, nt))  # Gaussian white noise on all floors

# Create an instance of newmark_beta with non-proportional damping
# M, K, C matrices are now constructed inside newmark_beta class
nm = newmark_beta(m, k, zeta_modal=zeta_modal, nt=nt, dt=dt, force=force)

# Print system information
print("Natural frequencies (Hz):", nm.w_n)
print("Modal damping ratios:", zeta_modal)

# For accurate results with non-proportional damping, we could modify newmark_beta
# or use the computed C matrix directly, but for now this approximation works

# Plot the acceleration responses for each floor over time (matching paper)
# Paper uses acceleration responses from virtual accelerometers
time = nm.t.flatten()  # Convert time vector to 1D array for plotting
plt.figure(figsize=(12, 6))
for i in range(len(m)):
    plt.plot(time[:1000], nm.a[i, :1000], label=f'Floor {i+1}')  # Show first 2.5 seconds
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Acceleration (m/s²)', fontsize=12)
plt.title('Acceleration Responses - Representative Time Histories', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate PSD for all floors
# Sampling frequency determined by dt
fs = 1 / dt

# Color scheme for different floors (matching ONLINE system)
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7']

# Store PSD results for all floors
psd_results = []
freq_all = None

# Estimate PSD using Welch's method for acceleration responses (matching paper)
# Paper analyzes acceleration responses from virtual accelerometers
# Welch method: divides signal into overlapping segments, computes FFT for each,
# and averages the power spectra to reduce variance
for floor_index in range(nDOF):
    acceleration = nm.a[floor_index, :]  # Use acceleration (matching paper)
    
    # Adjust nperseg for 100 Hz sampling rate
    # Use larger segment for better frequency resolution
    nperseg = 2048  # Segment length (20.48s at 100Hz)
    noverlap = nperseg // 2  # 50% overlap
    
    freq, psd = welch(acceleration, 
                      fs=fs, 
                      nperseg=nperseg,      # Segment length
                      noverlap=noverlap,    # 50% overlap (standard for Welch method)
                      window='hann',        # Hanning window to reduce spectral leakage
                      scaling='density',    # Return PSD (not power spectrum)
                      average='mean')       # Average method (default)
    
    # Convert PSD to dB scale; add a small value eps to avoid log(0)
    eps = 1e-12
    psd_db = 10 * np.log10(psd + eps)
    
    psd_results.append(psd_db)
    
    # Store frequency vector (same for all floors)
    if freq_all is None:
        freq_all = freq
    
    # Find the dominant frequency for each floor
    dominant_index = np.argmax(psd)
    dominant_frequency = freq[dominant_index]
    print(f"Floor {floor_index + 1} dominant frequency: {dominant_frequency:.4f} Hz")

# Plot the PSD for all floors: the x-axis is in linear scale
plt.figure(figsize=(12, 7))

# Plot PSD for each floor with different colors
for floor_index in range(nDOF):
    plt.plot(freq_all, psd_results[floor_index], 
             color=colors[floor_index], 
             linewidth=1.5, 
             label=f'Floor {floor_index + 1}')

plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('PSD (dB/Hz)', fontsize=12)
plt.title('Power Spectral Density (dB Scale) - Acceleration Responses', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(0, 50)  # Show up to 50 Hz (Nyquist frequency at 100 Hz sampling rate)
plt.legend(fontsize=9, ncol=2, loc='upper right')
plt.tight_layout()
plt.show()
