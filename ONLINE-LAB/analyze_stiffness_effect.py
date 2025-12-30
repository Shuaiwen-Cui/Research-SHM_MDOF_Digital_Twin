"""
Analyze the effect of stiffness increase on modal participation factors
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

def calculate_modes_and_participation(m, k):
    """Calculate natural frequencies and participation factors"""
    M = build_mass_matrix(m)
    K = build_stiffness_matrix(k)
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors = linalg.eigh(K, M)
    omega_n = np.sqrt(eigenvalues)
    f_n = omega_n / (2 * np.pi)
    
    # Sort by frequency
    sorted_indices = np.argsort(f_n)
    f_n_sorted = f_n[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    # Calculate participation factors for uniform excitation
    r = np.ones(nDOF)  # Uniform excitation
    participation_factors = []
    
    for i in range(nDOF):
        phi = eigenvectors_sorted[:, i]
        # Normalize to unit modal mass
        modal_mass = phi.T @ M @ phi
        phi_normalized = phi / np.sqrt(modal_mass)
        # Participation factor
        participation = phi_normalized.T @ M @ r
        participation_factors.append(participation)
    
    return f_n_sorted, participation_factors, sorted_indices

print("=" * 80)
print("Effect of Stiffness Increase on Modal Participation")
print("=" * 80)

# Different stiffness multipliers
stiffness_factors = [1.0, 1.5, 2.0, 3.0, 5.0]

# Store results
results = []

for factor in stiffness_factors:
    k_modified = k_initial * factor
    f_n, participation, sorted_indices = calculate_modes_and_participation(m_initial, k_modified)
    
    # Calculate contribution percentages
    total_contribution = np.sum(np.abs(participation))
    contribution_pct = [abs(p) / total_contribution * 100 for p in participation]
    
    results.append({
        'factor': factor,
        'frequencies': f_n,
        'participation': participation,
        'contribution_pct': contribution_pct,
        'sorted_indices': sorted_indices
    })

# Print comparison table
print("\n" + "=" * 80)
print("Natural Frequencies (Hz) vs Stiffness Factor")
print("=" * 80)
print(f"{'Stiffness':<12} {'Mode 1':<12} {'Mode 2':<12} {'Mode 3':<12} {'Mode 4':<12} {'Mode 5':<12}")
print("-" * 80)
for r in results:
    print(f"×{r['factor']:<11.1f} ", end="")
    for i in range(nDOF):
        print(f"{r['frequencies'][i]:>10.4f}  ", end="")
    print()

print("\n" + "=" * 80)
print("Modal Participation Contribution (%) vs Stiffness Factor")
print("=" * 80)
print(f"{'Stiffness':<12} {'Mode 1':<12} {'Mode 2':<12} {'Mode 3':<12} {'Mode 4':<12} {'Mode 5':<12}")
print("-" * 80)
for r in results:
    print(f"×{r['factor']:<11.1f} ", end="")
    for i in range(nDOF):
        print(f"{r['contribution_pct'][i]:>10.2f}% ", end="")
    print()

print("\n" + "=" * 80)
print("Change in Higher Mode Contribution (Mode 3-5 combined)")
print("=" * 80)
print(f"{'Stiffness':<12} {'Mode 1-2':<12} {'Mode 3-5':<12} {'Change':<12}")
print("-" * 80)

base_mode_3_5 = sum(results[0]['contribution_pct'][2:5])
for r in results:
    mode_1_2 = sum(r['contribution_pct'][0:2])
    mode_3_5 = sum(r['contribution_pct'][2:5])
    change = mode_3_5 - base_mode_3_5
    print(f"×{r['factor']:<11.1f} {mode_1_2:>10.2f}%  {mode_3_5:>10.2f}%  {change:>+10.2f}%")

print("\n" + "=" * 80)
print("Analysis:")
print("=" * 80)
print("""
Key Observations:

1. Frequency Scaling:
   - Natural frequencies scale with √(k/m)
   - Stiffness × N → Frequency × √N
   - Example: 2× stiffness → 1.414× frequency

2. Participation Factor Changes:
   - Stiffness increase changes the mode shapes
   - This affects how modes participate in response
   - Higher stiffness may redistribute energy among modes

3. Higher Mode Visibility:
   - If Mode 3-5 contribution increases → More visible in PSD
   - If Mode 3-5 contribution decreases → Less visible in PSD
   - The change depends on how mode shapes evolve

4. Practical Implications:
   - Stiffness increase generally increases all frequencies
   - But participation distribution may change
   - Need to check if higher modes become more prominent
""")

# Detailed analysis for one case
print("\n" + "=" * 80)
print("Detailed Comparison: Original vs 2× Stiffness")
print("=" * 80)
print(f"{'Mode':<8} {'Original Freq':<15} {'2× Stiff Freq':<15} {'Original %':<12} {'2× Stiff %':<12} {'Change':<10}")
print("-" * 80)

original = results[0]
doubled = results[2]  # factor = 2.0

for i in range(nDOF):
    mode_num = original['sorted_indices'][i] + 1
    orig_freq = original['frequencies'][i]
    new_freq = doubled['frequencies'][i]
    orig_pct = original['contribution_pct'][i]
    new_pct = doubled['contribution_pct'][i]
    change = new_pct - orig_pct
    
    print(f"Mode {mode_num:<6} {orig_freq:>12.4f} Hz  {new_freq:>12.4f} Hz  {orig_pct:>10.2f}%  {new_pct:>10.2f}%  {change:>+9.2f}%")

