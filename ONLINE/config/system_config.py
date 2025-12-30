"""
System configuration parameters
"""
import numpy as np

# Number of degrees of freedom
nDOF = 5

# Time step
dt = 0.02  # seconds

# Initial system parameters (can be updated by SYSTEM module)
# Masses (kg) for each floor (from bottom to top)
m_initial = np.array([1000, 800, 600, 500, 400])

# Stiffnesses (N/m) for each floor (5× increased for better higher mode visibility)
k_initial = np.array([250000, 200000, 150000, 125000, 100000])

# Damping ratio
zeta_initial = 0.05

# Newmark-Beta parameters
gamma_newmark = 0.5
beta_newmark = 0.25

