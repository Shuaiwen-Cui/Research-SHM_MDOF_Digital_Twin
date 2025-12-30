"""
System updater: provides system matrices M(t), K(t), C(t)
"""
import numpy as np
import time
import sys
import os
from scipy import linalg

# Add ONLINE directory to path
ONLINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ONLINE_DIR not in sys.path:
    sys.path.insert(0, ONLINE_DIR)

from shared.data_protocol import SystemData
from shared.communication import CommunicationChannels
from config.system_config import nDOF, dt, m_initial, k_initial, zeta_initial, zeta_modal_initial

class SystemUpdater:
    """Generates system matrices M(t), K(t), C(t)"""
    
    def __init__(self, channels: CommunicationChannels,
                 m: np.ndarray = None,
                 k: np.ndarray = None,
                 zeta: float = None,
                 zeta_modal: np.ndarray = None,
                 time_varying: bool = False):
        """
        Parameters:
        -----------
        channels: CommunicationChannels
            Communication channels for sending data
        m: np.ndarray
            Initial mass array (if None, use config default)
        k: np.ndarray
            Initial stiffness array (if None, use config default)
        zeta: float
            Initial damping ratio for proportional damping (if None, use config default)
        zeta_modal: np.ndarray
            Mode-specific damping ratios for non-proportional damping (if None, use config default)
        time_varying: bool
            Whether to generate time-varying system parameters
        """
        self.channels = channels
        self.current_time = 0.0
        self.time_varying = time_varying
        
        # Use provided parameters or defaults
        self.m = m if m is not None else m_initial.copy()
        self.k = k if k is not None else k_initial.copy()
        self.zeta = zeta if zeta is not None else zeta_initial
        self.zeta_modal = zeta_modal if zeta_modal is not None else zeta_modal_initial.copy()
        
        # Initialize matrices
        self.M = self._build_mass_matrix(self.m)
        self.K = self._build_stiffness_matrix(self.k)
        self.C = None  # Will be computed from zeta or zeta_modal
        
    def _build_mass_matrix(self, m: np.ndarray) -> np.ndarray:
        """Build diagonal mass matrix"""
        return np.diag(m)
    
    def _build_stiffness_matrix(self, k: np.ndarray) -> np.ndarray:
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
    
    def _build_damping_matrix(self, M: np.ndarray, K: np.ndarray, 
                              zeta: float = None, zeta_modal: np.ndarray = None) -> np.ndarray:
        """
        Build damping matrix - supports both proportional and non-proportional damping
        
        Parameters:
        -----------
        M: np.ndarray
            Mass matrix
        K: np.ndarray
            Stiffness matrix
        zeta: float, optional
            Damping ratio for proportional damping
        zeta_modal: np.ndarray, optional
            Mode-specific damping ratios for non-proportional damping
        
        Returns:
        --------
        C: np.ndarray
            Damping matrix
        """
        # Calculate natural frequencies and mode shapes
        eigenvalues, eigenvectors = linalg.eigh(K, M)
        omega_n = np.sqrt(eigenvalues)  # Natural frequencies (rad/s)
        
        if zeta_modal is not None:
            # Non-proportional modal damping
            # Normalize mode shapes to unit modal mass
            Phi = eigenvectors.copy()
            for i in range(len(Phi[0])):
                modal_mass = Phi[:, i].T @ M @ Phi[:, i]
                Phi[:, i] = Phi[:, i] / np.sqrt(modal_mass)
            
            # Build modal damping matrix
            Cn = np.diag(2 * zeta_modal * omega_n)  # Modal damping matrix
            
            # Convert to physical damping matrix: C = Φ^(-T) * Cn * Φ^(-1)
            # Since Phi is normalized: Phi^T * M * Phi = I, so Phi^(-1) = Phi^T * M
            Phi_inv = Phi.T @ M
            C = Phi_inv.T @ Cn @ Phi_inv  # Physical damping matrix
        else:
            # Proportional damping (Rayleigh damping)
            omega = omega_n[0]  # Fundamental circular frequency
            alpha = omega * zeta
            beta = zeta / omega
            C = alpha * M + beta * K
        
        return C
    
    def _update_system_parameters(self):
        """Update system parameters (can be time-varying)"""
        if self.time_varying:
            # Example: small random variations
            # In real application, this would come from damage detection, etc.
            self.m = m_initial * (1 + 0.01 * np.sin(self.current_time * 0.1))
            self.k = k_initial * (1 + 0.02 * np.sin(self.current_time * 0.15))
            # zeta can also vary
        else:
            # Keep initial values
            pass
        
        # Rebuild matrices
        self.M = self._build_mass_matrix(self.m)
        self.K = self._build_stiffness_matrix(self.k)
        self.C = self._build_damping_matrix(self.M, self.K, self.zeta, self.zeta_modal)
    
    def generate_system_data(self) -> SystemData:
        """Generate system matrices for current time"""
        self._update_system_parameters()
        
        return SystemData(
            timestamp=self.current_time,
            M=self.M.copy(),
            K=self.K.copy(),
            C=self.C.copy()
        )
    
    def run(self):
        """Main loop: generate and send system matrices"""
        print("System Updater started")
        
        while True:
            # Generate system data
            system_data = self.generate_system_data()
            self.channels.send_system(system_data)
            
            # Update time
            self.current_time += dt
            
            # Sleep to maintain real-time
            time.sleep(dt)

if __name__ == "__main__":
    channels = CommunicationChannels()
    updater = SystemUpdater(channels, time_varying=False)
    updater.run()

