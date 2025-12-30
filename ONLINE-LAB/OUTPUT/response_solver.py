"""
Real-time response solver using Newmark-Beta method
"""
import numpy as np
import time
import sys
import os

# Add ONLINE directory to path
ONLINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ONLINE_DIR not in sys.path:
    sys.path.insert(0, ONLINE_DIR)

from shared.data_protocol import ForceData, SystemData, ResponseData
from shared.communication import CommunicationChannels
from config.system_config import nDOF, dt, gamma_newmark, beta_newmark

class RealtimeSolver:
    """Real-time Newmark-Beta solver"""
    
    def __init__(self, channels: CommunicationChannels):
        """
        Parameters:
        -----------
        channels: CommunicationChannels
            Communication channels for receiving force/system and sending response
        """
        self.channels = channels
        
        # Initialize state vectors (at t-1)
        self.d_prev = np.zeros(nDOF)  # Displacement at t-1
        self.v_prev = np.zeros(nDOF)  # Velocity at t-1
        self.a_prev = np.zeros(nDOF)  # Acceleration at t-1
        
        # Current system matrices (will be updated)
        self.M = None
        self.K = None
        self.C = None
        
        # Newmark-Beta coefficients (constant)
        self.a0 = 1 / (beta_newmark * dt * dt)
        self.a1 = gamma_newmark / (beta_newmark * dt)
        self.a2 = 1 / (beta_newmark * dt)
        self.a3 = 1 / (2 * beta_newmark) - 1
        self.a4 = gamma_newmark / beta_newmark - 1
        self.a5 = dt / 2 * (gamma_newmark / beta_newmark - 2)
        self.a6 = dt * (1 - gamma_newmark)
        self.a7 = dt * gamma_newmark
        
        # Equivalent stiffness matrix (will be updated when system matrices change)
        self.ke = None
        self.ke_inv = None
        
        self.current_time = 0.0
        self.initialized = False
    
    def update_system_matrices(self, M: np.ndarray, K: np.ndarray, C: np.ndarray):
        """Update system matrices and recompute equivalent stiffness"""
        self.M = M.copy()
        self.K = K.copy()
        self.C = C.copy()
        
        # Recompute equivalent stiffness matrix
        self.ke = self.K + self.a0 * self.M + self.a1 * self.C
        self.ke_inv = np.linalg.inv(self.ke)
    
    def solve_step(self, force: np.ndarray) -> tuple:
        """
        Solve one time step using Newmark-Beta method
        
        Parameters:
        -----------
        force: np.ndarray
            Force vector at current time step t
            
        Returns:
        --------
        d, v, a: tuple of np.ndarray
            Displacement, velocity, acceleration at time t
        """
        if not self.initialized:
            raise ValueError("Solver not initialized. System matrices must be set first.")
        
        # Compute equivalent force
        fe = (np.reshape(force, [nDOF, 1]) +
              np.dot(self.M, np.reshape((self.a0 * self.d_prev +
                                         self.a2 * self.v_prev +
                                         self.a3 * self.a_prev), [nDOF, 1])) +
              np.dot(self.C, np.reshape((self.a1 * self.d_prev +
                                         self.a4 * self.v_prev +
                                         self.a5 * self.a_prev), [nDOF, 1])))
        
        # Solve for displacement
        d = np.reshape(np.dot(self.ke_inv, fe), [nDOF])
        
        # Compute acceleration
        a = self.a0 * (d - self.d_prev) - self.a2 * self.v_prev - self.a3 * self.a_prev
        
        # Compute velocity
        v = self.v_prev + self.a6 * self.a_prev + self.a7 * a
        
        # Update previous state for next iteration
        self.d_prev = d.copy()
        self.v_prev = v.copy()
        self.a_prev = a.copy()
        
        return d, v, a
    
    def run(self):
        """Main loop: receive inputs and compute responses"""
        print("Response Solver started")
        
        # Wait for initial system matrices
        print("Waiting for initial system matrices...")
        system_data = None
        while system_data is None:
            system_data = self.channels.receive_system(timeout=0.1)
        
        # Initialize system matrices
        self.update_system_matrices(system_data.M, system_data.K, system_data.C)
        self.initialized = True
        self.current_time = system_data.timestamp
        print("System matrices initialized")
        
        # Main loop
        while True:
            # Receive force and system data (non-blocking)
            force_data = self.channels.receive_force(timeout=0.01)
            system_data = self.channels.receive_system(timeout=0.01)
            
            # Update system matrices if new data available
            if system_data is not None:
                self.update_system_matrices(system_data.M, system_data.K, system_data.C)
                self.current_time = system_data.timestamp
            
            # Process force if available
            if force_data is not None:
                # Solve for response
                d, v, a = self.solve_step(force_data.force)
                
                # Create and send response data
                response_data = ResponseData(
                    timestamp=force_data.timestamp,
                    displacement=d,
                    velocity=v,
                    acceleration=a
                )
                self.channels.send_response(response_data)
                
                self.current_time = force_data.timestamp

if __name__ == "__main__":
    channels = CommunicationChannels()
    solver = RealtimeSolver(channels)
    solver.run()

