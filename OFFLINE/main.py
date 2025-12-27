import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Import modules from the same directory
from newmark_beta import newmark_beta
from ambient_vibration import ambient_vibration_gen
from earthquake import spec_density, earthquake_gen
from impact import impact_gen
from strongwind import strong_wind_gen

# Define system parameters for a 3-story building
m = np.array([1000, 800, 600])        # Masses (kg) for each floor (from bottom to top)
k = np.array([50000, 40000, 30000])     # Stiffnesses (N/m) for each floor
zeta = 0.05                           # Damping ratio
nt = 12000                            # Total number of time steps
dt = 0.01                             # Time increment (s)

# Create an external force matrix with shape (3, nt)
# For example, apply a force of 1000 N on the top floor between time steps 10 and 20
force = np.zeros((3, nt))
force[2, 10:21] = 1000

# Create an instance of newmark_beta with the given parameters
nm = newmark_beta(m, k, zeta, nt, dt, force)

# Plot the displacement responses for each floor over time
time = nm.t.flatten()  # Convert time vector to 1D array for plotting
plt.figure()
for i in range(len(m)):
    plt.plot(time, nm.d[i, :], label=f'Floor {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Floor Displacements Over Time')
plt.legend()
plt.show()

# Select the floor data to analyze (0 for the first floor, 1 for the second, etc.)
floor_index = 0  # Analyze the first floor

# Extract the displacement data for the selected floor
displacement = nm.d[floor_index, :]

# Sampling frequency determined by dt
fs = 1 / dt

# Estimate PSD using Welch's method; nperseg can be adjusted based on data length
freq, psd = welch(displacement, fs=fs, nperseg=1024)

# Convert PSD to dB scale; add a small value eps to avoid log(0)
eps = 1e-12
psd_db = 10 * np.log10(psd + eps)

# Find the dominant frequency by locating the frequency corresponding to the maximum PSD
dominant_index = np.argmax(psd)
dominant_frequency = freq[dominant_index]
print("The dominant frequency of floor", floor_index + 1, "is:", dominant_frequency, "Hz")

# Plot the PSD: the x-axis is in log scale
plt.figure()
plt.semilogx(freq, psd_db)  # Use semilogx to convert the x-axis to a logarithmic scale
plt.xlabel('Frequency (Hz) [Log Scale]')
plt.ylabel('PSD (dB/Hz)')
plt.title('Power Spectral Density (dB Scale) - Floor ' + str(floor_index + 1))
plt.show()
