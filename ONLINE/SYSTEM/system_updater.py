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
from config.system_config import nDOF, dt, m_initial, k_initial, zeta_initial

class SystemUpdater:
    """Generates system matrices M(t), K(t), C(t)"""
    
    def __init__(self, channels: CommunicationChannels,
                 m: np.ndarray = None,
                 k: np.ndarray = None,
                 zeta: float = None,
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
            Initial damping ratio (if None, use config default)
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
        
        # Initialize matrices
        self.M = self._build_mass_matrix(self.m)
        self.K = self._build_stiffness_matrix(self.k)
        self.C = None  # Will be computed from zeta
        
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
    
    def _build_damping_matrix(self, M: np.ndarray, K: np.ndarray, zeta: float) -> np.ndarray:
        """Build proportional damping matrix"""
        # Calculate natural frequencies
        w_n = 1 / np.sqrt(linalg.eigvals(M, K).real) / (2 * np.pi)
        omiga = w_n[0] * 2 * np.pi  # Fundamental circular frequency
        alpha = omiga * zeta
        beta = zeta / omiga
        return alpha * M + beta * K
    
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
        self.C = self._build_damping_matrix(self.M, self.K, self.zeta)
    
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

