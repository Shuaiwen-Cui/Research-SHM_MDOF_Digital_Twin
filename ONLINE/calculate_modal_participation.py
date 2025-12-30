"""
Calculate modal participation factors for the 5DOF structure
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

# Solve generalized eigenvalue problem
eigenvalues, eigenvectors = linalg.eigh(K, M)

# Natural frequencies
omega_n = np.sqrt(eigenvalues)
f_n = omega_n / (2 * np.pi)

# Sort by frequency
sorted_indices = np.argsort(f_n)
f_n_sorted = f_n[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

print("=" * 70)
print("Modal Participation Factors (模态参与因子)")
print("=" * 70)

# Different excitation patterns
excitation_patterns = {
    "Uniform (所有楼层同时激励)": np.ones(nDOF),
    "Base excitation (底部激励)": np.array([1, 0, 0, 0, 0]),
    "Top excitation (顶部激励)": np.array([0, 0, 0, 0, 1]),
    "Middle excitation (中间激励)": np.array([0, 0, 1, 0, 0]),
}

for pattern_name, r in excitation_patterns.items():
    print(f"\n{pattern_name}:")
    print("-" * 70)
    print(f"{'Mode':<6} {'Frequency (Hz)':<18} {'Participation':<20} {'% Contribution':<15}")
    print("-" * 70)
    
    participation_factors = []
    for i in range(nDOF):
        mode_idx = sorted_indices[i]
        phi = eigenvectors_sorted[:, i]
        
        # Normalize to unit modal mass
        modal_mass = phi.T @ M @ phi
        phi_normalized = phi / np.sqrt(modal_mass)
        
        # Calculate participation factor: Γ = φ^T * M * r
        # For unit modal mass normalization: Γ = φ^T * M * r
        participation = phi_normalized.T @ M @ r
        participation_factors.append(participation)
    
    # Calculate total contribution
    total_contribution = np.sum(np.abs(participation_factors))
    
    for i in range(nDOF):
        mode_idx = sorted_indices[i]
        freq = f_n_sorted[i]
        gamma = participation_factors[i]
        contribution_pct = (np.abs(gamma) / total_contribution * 100) if total_contribution > 0 else 0
        
        print(f"Mode {mode_idx + 1:<5} {freq:>12.4f} Hz    {gamma:>15.6f}    {contribution_pct:>10.2f}%")

print("\n" + "=" * 70)
print("Explanation:")
print("=" * 70)
print("""
Modal Participation Factor (Γ) indicates how much each mode contributes to the response.

Key points:
1. Higher |Γ| = More contribution to response
2. Sign indicates direction (positive/negative)
3. Sum of |Γ| gives total modal contribution
4. Different excitation patterns excite different modes differently

For PSD analysis:
- Modes with high participation factors → Clear peaks in PSD
- Modes with low participation factors → Weak or invisible peaks
- This explains why Mode 3-5 peaks are less visible than Mode 1-2
""")

