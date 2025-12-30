"""
System configuration parameters
Matching SYSID/main.py parameters
"""
import numpy as np

# Number of degrees of freedom
nDOF = 5

# Time step - 100 Hz sampling rate (doubled)
dt = 1.0 / 100.0  # seconds (100 Hz sampling rate)

# Initial system parameters (can be updated by SYSTEM module)
# Masses (kg) for each floor (from bottom to top) - matching SYSID
m_initial = np.array([700, 700, 700, 700, 700])

# Stiffnesses (N/m) for each floor - doubled
k_initial = np.array([1000000, 1000000, 1000000, 1000000, 1000000])  # 100 kN/m = 100000 N/m

# Non-proportional modal damping: mode-specific damping ratios (matching SYSID)
zeta_modal_initial = np.array([0.02, 0.02, 0.02, 0.02, 0.02])  # Mode-specific damping ratios

# Proportional damping ratio (for backward compatibility, using average of modal damping)
zeta_initial = np.mean(zeta_modal_initial)  # Average damping ratio

# Newmark-Beta parameters
gamma_newmark = 0.5
beta_newmark = 0.25

# PSD calculation parameters
psd_lowpass_cutoff = 15.0  # Low-pass filter cutoff frequency (Hz) for PSD calculation

