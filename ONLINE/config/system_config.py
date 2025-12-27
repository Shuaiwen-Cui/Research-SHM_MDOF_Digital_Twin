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

# Stiffnesses (N/m) for each floor
k_initial = np.array([50000, 40000, 30000, 25000, 20000])

# Damping ratio
zeta_initial = 0.05

# Newmark-Beta parameters
gamma_newmark = 0.5
beta_newmark = 0.25

