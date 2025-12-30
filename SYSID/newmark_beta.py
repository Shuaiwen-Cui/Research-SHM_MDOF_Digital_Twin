import numpy as np
from scipy import linalg

class newmark_beta:
    """
    ============================================================================
    NEWMARK-BETA TIME INTEGRATION METHOD FOR MDOF STRUCTURAL DYNAMICS
    ============================================================================
    
    Algorithm Overview:
    -------------------
    This class implements the Newmark-Beta implicit time integration method for
    solving the multi-degree-of-freedom (MDOF) dynamic equation of motion:
    
        M·ü(t) + C·u̇(t) + K·u(t) = F(t)
    
    where:
        M = mass matrix
        C = damping matrix
        K = stiffness matrix
        u, u̇, ü = displacement, velocity, acceleration
        F(t) = external force vector
    
    Mathematical Formulation:
    ------------------------
    The Newmark-Beta method uses two key assumptions:
        u̇(t+Δt) = u̇(t) + [(1-γ)·ü(t) + γ·ü(t+Δt)]·Δt
        u(t+Δt) = u(t) + u̇(t)·Δt + [(0.5-β)·ü(t) + β·ü(t+Δt)]·Δt²
    
    Standard parameters: γ=0.5, β=0.25 (average acceleration method - unconditionally stable)
    
    System Matrices:
    ---------------
    - Mass Matrix (M): Diagonal matrix for lumped mass model
    - Stiffness Matrix (K): Tridiagonal for shear building model
    - Damping Matrix (C): Proportional (Rayleigh) or Modal damping
    
    State Variables:
    ---------------
        self.m_matrax: Mass matrix [n×n]
        self.k_matrix: Stiffness matrix [n×n]
        self.c_matrax: Damping matrix [n×n]
        self.w_n: Natural frequencies [Hz]
        self.omega_n: Circular frequencies [rad/s]
        self.Phi: Mass-normalized mode shapes
        self.t: Time vector [1×(nt+1)]
        self.d: Displacement history [n×(nt+1)]
        self.v: Velocity history [n×(nt+1)]
        self.a: Acceleration history [n×(nt+1)]
    """
    
    def __init__(self, m, k, zeta=None, zeta_modal=None, nt=None, dt=None, force=None, gama_newmark=0.5, beta_newmark=0.25):
        """
        Initialize Newmark-Beta time integration solver.
        
        Parameters:
        -----------
        m : list or array
            Mass values for each floor [m1, m2, ..., mn] (bottom to top)
        k : list or array
            Inter-story stiffness values [k1, k2, ..., kn] (bottom to top)
        zeta : float, optional
            Uniform damping ratio for Rayleigh proportional damping (e.g., 0.05 for 5%)
        zeta_modal : array, optional
            Mode-specific damping ratios [ζ1, ζ2, ..., ζn] for modal damping
        nt : int, optional
            Total number of time steps
        dt : float, optional
            Time step size [seconds]
        force : ndarray, optional
            External force matrix [n×nt] where force[i,j] = force on floor i at time j
        gama_newmark : float, default=0.5
            Newmark-β parameter γ (controls numerical damping)
        beta_newmark : float, default=0.25
            Newmark-β parameter β (controls accuracy and stability)
            
        Algorithm Flow:
        ---------------
        1. Validate inputs and construct system matrices (M, K, C)
        2. Perform modal analysis (eigenvalues & eigenvectors)
        3. Normalize mode shapes and build damping matrix
        4. Compute Newmark-β integration coefficients
        5. Execute time-stepping integration loop
        """
        
        # ========================================================================
        # STEP 1: INPUT VALIDATION
        # ========================================================================
        if len(m) != len(k):
            print('The lengths of mass and stiffness lists are different.')
            quit()

        num = len(k)  # Number of DOFs (floors)
        
        # ========================================================================
        # STEP 2: CONSTRUCT SYSTEM MATRICES
        # ========================================================================
        
        # Build mass matrix (diagonal for lumped mass model)
        self.m_matrax = self._build_mass_matrix(m)
        
        # Build stiffness matrix (tridiagonal for shear building)
        self.k_matrix = self._build_stiffness_matrix(k)
        
        # ========================================================================
        # STEP 3: MODAL ANALYSIS (Eigenvalue Problem)
        # ========================================================================
        # Solve generalized eigenvalue problem: K·Φ = M·Φ·Λ
        # where Λ = diag(ω₁², ω₂², ..., ωₙ²)
        
        eigenvalues, eigenvectors = linalg.eigh(self.k_matrix, self.m_matrax)
        self.w_n = np.sqrt(eigenvalues) / (2 * np.pi)  # Natural frequencies [Hz]
        self.omega_n = self.w_n * 2 * np.pi  # Circular frequencies [rad/s]
        self.eigenvectors = eigenvectors  # Raw mode shapes
        
        # ========================================================================
        # STEP 4: NORMALIZE MODE SHAPES
        # ========================================================================
        # Normalize to satisfy: Φᵀ·M·Φ = I (identity matrix)
        self.Phi = self._normalize_mode_shapes(eigenvectors)
        
        # ========================================================================
        # STEP 5: BUILD DAMPING MATRIX
        # ========================================================================
        # Two options:
        # 1. Modal damping: Different damping ratio for each mode
        # 2. Rayleigh damping: C = α·M + β·K (proportional damping)
        
        if zeta_modal is not None:
            # Non-proportional modal damping
            self.c_matrax = self._build_nonproportional_damping(zeta_modal)
            self.zeta_modal = zeta_modal
        elif zeta is not None:
            # Proportional Rayleigh damping
            self.c_matrax = self._build_proportional_damping(zeta)
            self.zeta = zeta
        else:
            raise ValueError("Either zeta or zeta_modal must be provided")

        # ========================================================================
        # STEP 6: COMPUTE NEWMARK-BETA INTEGRATION COEFFICIENTS
        # ========================================================================
        # These coefficients transform the implicit equation into an efficient
        # recurrence relation for time-stepping
        
        a0 = 1 / (beta_newmark * dt * dt)
        a1 = gama_newmark / (beta_newmark * dt)
        a2 = 1 / (beta_newmark * dt)
        a3 = 1 / (2 * beta_newmark) - 1
        a4 = gama_newmark / beta_newmark - 1
        a5 = dt / 2 * (gama_newmark / beta_newmark - 2)
        a6 = dt * (1 - gama_newmark)
        a7 = dt * gama_newmark

        # ========================================================================
        # STEP 7: INITIALIZE STATE ARRAYS
        # ========================================================================
        # Initial conditions: All DOFs at rest (zero displacement, velocity, acceleration)
        
        self.t = np.zeros([1, nt + 1])  # Time vector
        self.d = np.zeros([num, nt + 1])  # Displacement history [m]
        self.v = np.zeros([num, nt + 1])  # Velocity history [m/s]
        self.a = np.zeros([num, nt + 1])  # Acceleration history [m/s²]

        # ========================================================================
        # STEP 8: COMPUTE EFFECTIVE STIFFNESS MATRIX
        # ========================================================================
        # Transform implicit equation to: K_eff·u(t+Δt) = F_eff(t)
        # where K_eff = K + a0·M + a1·C (constant throughout integration)
        
        ke = self.k_matrix + a0 * self.m_matrax + a1 * self.c_matrax

        # ========================================================================
        # STEP 9: TIME-STEPPING INTEGRATION LOOP
        # ========================================================================
        # At each time step:
        #   1. Compute effective force F_eff incorporating previous states
        #   2. Solve for displacement: u(t+Δt) = K_eff⁻¹·F_eff
        #   3. Update velocity and acceleration using Newmark formulas
        
        if force is not None and nt is not None and dt is not None:
            for j in range(1, nt + 1):
                # Update time
                self.t[0][j] = j * dt
                
                # Compute effective force vector
                # F_eff = F(t) + M·(a0·u + a2·u̇ + a3·ü) + C·(a1·u + a4·u̇ + a5·ü)
                fe = (np.reshape(force[:, j - 1], [num, 1]) +
                      np.dot(self.m_matrax, np.reshape((a0 * self.d[:, j - 1] +
                                                         a2 * self.v[:, j - 1] +
                                                         a3 * self.a[:, j - 1]), [num, 1])) +
                      np.dot(self.c_matrax, np.reshape((a1 * self.d[:, j - 1] +
                                                         a4 * self.v[:, j - 1] +
                                                         a5 * self.a[:, j - 1]), [num, 1])))
                
                # Solve for displacement at next time step
                self.d[:, j] = np.reshape(np.dot(np.linalg.inv(ke), fe), [1, num])
                
                # Update acceleration using Newmark formula
                self.a[:, j] = a0 * (self.d[:, j] - self.d[:, j - 1]) - a2 * self.v[:, j - 1] - a3 * self.a[:, j - 1]
                
                # Update velocity using Newmark formula
                self.v[:, j] = self.v[:, j - 1] + a6 * self.a[:, j - 1] + a7 * self.a[:, j]
    
    def _build_mass_matrix(self, m):
        """
        Build diagonal mass matrix for lumped mass model.
        
        Mathematical Form:
        -----------------
        M = diag(m₁, m₂, ..., mₙ)
        
        Parameters:
        -----------
        m : array-like
            Mass values for each DOF
            
        Returns:
        --------
        M : ndarray
            Diagonal mass matrix [n×n]
        """
        return np.diag(m)
    
    def _build_stiffness_matrix(self, k):
        """
        Build stiffness matrix for shear building (cantilever) structure.
        
        Mathematical Form (Tridiagonal):
        --------------------------------
        K = [  k₁+k₂   -k₂      0     ...  ]
            [  -k₂     k₂+k₃   -k₃    ...  ]
            [   0      -k₃     k₃+k₄  ...  ]
            [  ...     ...     ...    kₙ   ]
        
        Physical Interpretation:
        -----------------------
        - Diagonal: Sum of stiffnesses connected to that floor
        - Off-diagonal: Negative inter-story stiffness (coupling)
        - Top floor: Only one stiffness (last story)
        
        Parameters:
        -----------
        k : array-like
            Inter-story stiffness values [k₁, k₂, ..., kₙ]
            
        Returns:
        --------
        K : ndarray
            Stiffness matrix [n×n]
        """
        num = len(k)
        k_matrix = np.zeros((num, num))
        for i in range(num):
            if i == 0:
                # Bottom floor: connected to ground and floor above
                k_matrix[i][i] = k[0] + k[1]
                k_matrix[i][i+1] = -k[1]
            elif i == num - 1:
                # Top floor: only connected to floor below
                k_matrix[i][i] = k[-1]
                k_matrix[i][i-1] = -k[-1]
            else:
                # Middle floors: connected to floors above and below
                k_matrix[i][i-1] = -k[i]
                k_matrix[i][i] = k[i] + k[i+1]
                k_matrix[i][i+1] = -k[i+1]
        return k_matrix
    
    def _normalize_mode_shapes(self, eigenvectors):
        """
        Normalize mode shapes to unit modal mass.
        
        Mathematical Formulation:
        ------------------------
        For each mode i, normalize such that:
            Φᵢᵀ · M · Φᵢ = 1
        
        This ensures modal orthogonality:
            Φᵀ · M · Φ = I (identity matrix)
            Φᵀ · K · Φ = diag(ω₁², ω₂², ..., ωₙ²)
        
        Parameters:
        -----------
        eigenvectors : ndarray
            Raw mode shapes from eigenvalue solver [n×n]
            
        Returns:
        --------
        Phi : ndarray
            Mass-normalized mode shapes [n×n]
        """
        Phi = eigenvectors.copy()
        for i in range(len(Phi[0])):
            # Compute modal mass: mᵢ = Φᵢᵀ·M·Φᵢ
            modal_mass = Phi[:, i].T @ self.m_matrax @ Phi[:, i]
            # Normalize: Φᵢ = Φᵢ / √mᵢ
            Phi[:, i] = Phi[:, i] / np.sqrt(modal_mass)
        return Phi
    
    def _build_proportional_damping(self, zeta):
        """
        Build Rayleigh proportional damping matrix.
        
        Mathematical Formulation:
        ------------------------
        C = α·M + β·K (Rayleigh damping)
        
        where:
            α = ζ·ω₁  (mass-proportional coefficient)
            β = ζ/ω₁  (stiffness-proportional coefficient)
            ω₁ = fundamental circular frequency [rad/s]
        
        This formulation ensures that the fundamental mode has the
        specified damping ratio ζ.
        
        Parameters:
        -----------
        zeta : float
            Target damping ratio (e.g., 0.05 for 5% damping)
            
        Returns:
        --------
        C : ndarray
            Rayleigh damping matrix [n×n]
        """
        omega = self.omega_n[0]  # Fundamental circular frequency [rad/s]
        alpha = omega * zeta     # Mass-proportional coefficient
        beta = zeta / omega      # Stiffness-proportional coefficient
        return alpha * self.m_matrax + beta * self.k_matrix
    
    def _build_nonproportional_damping(self, zeta_modal):
        """
        Build non-proportional damping matrix from mode-specific damping ratios.
        
        Mathematical Formulation:
        ------------------------
        Modal damping matrix (diagonal):
            Cₙ = diag(2ζ₁ω₁, 2ζ₂ω₂, ..., 2ζₙωₙ)
        
        Transform to physical coordinates:
            C = Φ⁻ᵀ · Cₙ · Φ⁻¹
        
        Using modal orthogonality (Φᵀ·M·Φ = I):
            Φ⁻¹ = Φᵀ·M
            Φ⁻ᵀ = M·Φ
        
        Therefore:
            C = (M·Φ) · Cₙ · (Φᵀ·M)
        
        Advantages:
        -----------
        - Allows different damping ratios for each mode
        - Useful for structures with mode-dependent energy dissipation
        - More realistic than Rayleigh damping for complex structures
        
        Parameters:
        -----------
        zeta_modal : array-like
            Damping ratio for each mode [ζ₁, ζ₂, ..., ζₙ]
            
        Returns:
        --------
        C : ndarray
            Non-proportional damping matrix [n×n]
        """
        # Build modal damping matrix (diagonal)
        Cn = np.diag(2 * zeta_modal * self.omega_n)
        
        # Transform to physical coordinates
        # Φ⁻¹ = Φᵀ·M (using modal orthogonality)
        Phi_inv = self.Phi.T @ self.m_matrax
        
        # C = Φ⁻ᵀ · Cₙ · Φ⁻¹
        C = Phi_inv.T @ Cn @ Phi_inv
        
        return C