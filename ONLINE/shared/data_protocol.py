"""
Data protocol definitions for communication between modules
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ForceData:
    """Force vector data at time t"""
    timestamp: float
    force: np.ndarray  # Shape: (nDOF,)
    
@dataclass
class SystemData:
    """System matrices at time t"""
    timestamp: float
    M: np.ndarray  # Mass matrix (nDOF x nDOF)
    K: np.ndarray  # Stiffness matrix (nDOF x nDOF)
    C: np.ndarray  # Damping matrix (nDOF x nDOF)
    
@dataclass
class ResponseData:
    """Response data at time t"""
    timestamp: float
    displacement: np.ndarray  # Shape: (nDOF,)
    velocity: np.ndarray      # Shape: (nDOF,)
    acceleration: np.ndarray  # Shape: (nDOF,)

