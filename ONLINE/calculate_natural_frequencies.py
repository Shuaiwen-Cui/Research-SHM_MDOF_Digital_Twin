"""
Calculate natural frequencies of the 5DOF structure
"""
import numpy as np
from scipy import linalg
import sys
import os

# Add ONLINE directory to path
ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
if ONLINE_DIR not in sys.path:
    sys.path.insert(0, ONLINE_DIR)

from config.system_config import nDOF, m_initial, k_initial

def build_mass_matrix(m):
    """Build diagonal mass matrix"""
    return np.diag(m)

def build_stiffness_matrix(k):
    """Build stiffness matrix for cantilever structure"""
    num = len(k)
    k_matrix = np.zeros((num, num))
    for i in range(num):
        if i == 0:
            k_matrix[i][i] = k[0] + k[1]
            k_matrix[i][i+1] = -k[1]
        elif i == num - 1:
            k_matrix[i][i] = k[-1]
            k_matrix[i][i-1] = -k[-1]
        else:
            k_matrix[i][i-1] = -k[i]
            k_matrix[i][i] = k[i] + k[i+1]
            k_matrix[i][i+1] = -k[i+1]
    return k_matrix

# Build matrices
M = build_mass_matrix(m_initial)
K = build_stiffness_matrix(k_initial)

print("=" * 60)
print("5DOF Structure Natural Frequencies Calculation")
print("=" * 60)
print(f"\nMass matrix M (kg):")
print(M)
print(f"\nStiffness matrix K (N/m):")
print(K)

# Solve generalized eigenvalue problem: K * φ = ω² * M * φ
# This gives: (K - ω² * M) * φ = 0
# Using scipy.linalg.eigh for symmetric matrices
eigenvalues, eigenvectors = linalg.eigh(K, M)

# Natural frequencies in rad/s
omega_n = np.sqrt(eigenvalues)

# Natural frequencies in Hz
f_n = omega_n / (2 * np.pi)

# Sort by frequency (ascending)
sorted_indices = np.argsort(f_n)
f_n_sorted = f_n[sorted_indices]
omega_n_sorted = omega_n[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

print("\n" + "=" * 60)
print("RESULTS: Natural Frequencies (固有频率)")
print("=" * 60)
print(f"\n{'Mode':<6} {'Frequency (Hz)':<18} {'Period (s)':<15} {'ω (rad/s)':<15}")
print("-" * 60)

for i in range(nDOF):
    mode_num = sorted_indices[i] + 1
    freq = f_n_sorted[i]
    period = 1.0 / freq
    omega = omega_n_sorted[i]
    print(f"Mode {mode_num:<5} {freq:>12.4f} Hz    {period:>10.4f} s    {omega:>12.4f} rad/s")

print("\n" + "=" * 60)
print("Mode Shapes (振型) - Normalized to unit modal mass")
print("=" * 60)

for i in range(nDOF):
    mode_idx = sorted_indices[i]
    mode_shape = eigenvectors_sorted[:, i]
    # Normalize to unit modal mass: φ^T * M * φ = 1
    modal_mass = mode_shape.T @ M @ mode_shape
    mode_shape_normalized = mode_shape / np.sqrt(modal_mass)
    print(f"\nMode {mode_idx + 1} (f = {f_n_sorted[i]:.4f} Hz):")
    print("Floor displacements (normalized):")
    for floor in range(nDOF):
        print(f"  Floor {floor + 1}: {mode_shape_normalized[floor]:>10.6f}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Fundamental frequency (1st mode): {f_n_sorted[0]:.4f} Hz")
print(f"Highest frequency (5th mode):    {f_n_sorted[-1]:.4f} Hz")
print(f"Frequency ratio (5th/1st):       {f_n_sorted[-1]/f_n_sorted[0]:.4f}")

