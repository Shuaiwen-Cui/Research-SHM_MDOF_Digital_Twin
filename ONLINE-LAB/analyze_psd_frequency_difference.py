"""
分析PSD测量频率与理论频率差异的原因
"""
import numpy as np
from scipy import linalg
from config.system_config import m_initial, k_initial, zeta_modal_initial

print("=" * 70)
print("PSD测量频率与理论频率差异分析")
print("=" * 70)

# 构建系统矩阵
M = np.diag(m_initial)
nDOF = len(k_initial)
K = np.zeros((nDOF, nDOF))
for i in range(nDOF):
    if i == 0:
        K[i][i] = k_initial[0] + k_initial[1]
        K[i][i+1] = -k_initial[1]
    elif i == nDOF - 1:
        K[i][i] = k_initial[-1]
        K[i][i-1] = -k_initial[-1]
    else:
        K[i][i-1] = -k_initial[i]
        K[i][i] = k_initial[i] + k_initial[i+1]
        K[i][i+1] = -k_initial[i+1]

# 计算理论无阻尼频率
eigenvalues, eigenvectors = linalg.eigh(K, M)
omega_n = np.sqrt(eigenvalues)
f_n = omega_n / (2 * np.pi)
f_n_sorted = np.sort(f_n)

print("\n1. 理论无阻尼自然频率:")
for i in range(nDOF):
    print(f"   Mode {i+1}: {f_n_sorted[i]:.4f} Hz")

# 归一化振型
Phi = eigenvectors.copy()
for i in range(nDOF):
    modal_mass = Phi[:, i].T @ M @ Phi[:, i]
    Phi[:, i] = Phi[:, i] / np.sqrt(modal_mass)

# 构建非比例阻尼矩阵
Cn = np.diag(2 * zeta_modal_initial * omega_n)
Phi_inv = Phi.T @ M
C = Phi_inv.T @ Cn @ Phi_inv

# 计算有阻尼频率（非比例阻尼系统）
# 对于非比例阻尼，需要求解复特征值问题
# (λ²M + λC + K)φ = 0
# 简化为状态空间形式: [0 I; -M⁻¹K -M⁻¹C] [u; u̇] = λ[u; u̇]

M_inv = np.linalg.inv(M)
A11 = np.zeros((nDOF, nDOF))
A12 = np.eye(nDOF)
A21 = -M_inv @ K
A22 = -M_inv @ C
A = np.block([[A11, A12], [A21, A22]])

# 求解复特征值
eigenvals_complex, eigenvecs_complex = linalg.eig(A)

# 提取有阻尼频率（只取稳定模态，即实部为负的）
stable_indices = np.where(np.real(eigenvals_complex) < 0)[0]
eigenvals_stable = eigenvals_complex[stable_indices]
damped_freqs = np.abs(np.imag(eigenvals_stable)) / (2 * np.pi)
damped_freqs_sorted = np.sort(damped_freqs)

print("\n2. 有阻尼频率（非比例阻尼）:")
for i in range(min(nDOF, len(damped_freqs_sorted))):
    diff = damped_freqs_sorted[i] - f_n_sorted[i]
    diff_pct = (diff / f_n_sorted[i]) * 100
    print(f"   Mode {i+1}: {damped_freqs_sorted[i]:.4f} Hz (理论无阻尼: {f_n_sorted[i]:.4f} Hz, 差异: {diff:+.4f} Hz, {diff_pct:+.2f}%)")

# PSD参数分析
fs = 100.0
nperseg = 2048
df = fs / nperseg

print("\n3. PSD计算参数:")
print(f"   采样频率: {fs} Hz")
print(f"   分段长度 (nperseg): {nperseg}")
print(f"   频率分辨率 (df): {df:.4f} Hz")
print(f"   分段时长: {nperseg/fs:.2f} 秒")

# 分析频率分辨率对峰值定位的影响
print("\n4. 频率分辨率对峰值定位的影响:")
for i in range(nDOF):
    bins_per_peak = f_n_sorted[i] / df
    print(f"   Mode {i+1} ({f_n_sorted[i]:.4f} Hz): 约 {bins_per_peak:.1f} 个频率点/峰值")
    if bins_per_peak < 5:
        print(f"      ⚠️  分辨率可能不足，峰值可能被模糊")

# 模态参与度分析（假设均匀激励）
print("\n5. 模态参与度分析（均匀激励）:")
# 模态参与因子: Γ = Φ^T * M * r，其中r是激励分布向量
r_uniform = np.ones(nDOF)  # 均匀激励
modal_participation = np.zeros(nDOF)
for i in range(nDOF):
    modal_participation[i] = Phi[:, i].T @ M @ r_uniform

# 归一化参与度
modal_participation_norm = np.abs(modal_participation) / np.max(np.abs(modal_participation))

print("   模态参与度（归一化）:")
for i in range(nDOF):
    print(f"   Mode {i+1}: {modal_participation_norm[i]:.4f} ({modal_participation_norm[i]*100:.1f}%)")

# 阻尼对峰值的影响
print("\n6. 阻尼对PSD峰值的影响:")
for i in range(nDOF):
    zeta = zeta_modal_initial[i]
    # 峰值宽度（半功率带宽）: Δf ≈ 2*ζ*f_n
    bandwidth = 2 * zeta * f_n_sorted[i]
    print(f"   Mode {i+1}: 阻尼比={zeta:.3f}, 峰值带宽≈{bandwidth:.4f} Hz")
    if bandwidth < df:
        print(f"      ⚠️  峰值带宽小于频率分辨率，峰值可能不明显")

print("\n7. 可能的原因总结:")
print("   a) 非比例阻尼导致模态耦合，使PSD峰值位置偏离理论无阻尼频率")
print("   b) 高阶模态的阻尼比不同，导致频率偏移")
print("   c) 频率分辨率可能对高频模态不够精细")
print("   d) 模态参与度差异导致某些模态的峰值不明显")
print("   e) PSD平滑处理（3点移动平均）可能模糊峰值位置")
print("   f) 数据长度和分段数可能影响统计精度")

print("\n8. 建议改进措施:")
print("   - 增加nperseg以提高频率分辨率（如4096或8192）")
print("   - 减少或移除平滑处理以保持峰值清晰度")
print("   - 增加数据采集时间以获得更多分段")
print("   - 使用更精确的峰值检测算法（如抛物线插值）")
print("   - 考虑使用有阻尼频率作为理论值进行比较")

