"""
Communication interface for inter-module data exchange
"""
import queue
import multiprocessing
from typing import Optional
from .data_protocol import ForceData, SystemData, ResponseData

class CommunicationChannels:
    """Manages communication channels between modules"""
    
    def __init__(self, use_multiprocessing=False):
        """
        Parameters:
        -----------
        use_multiprocessing: bool
            If True, use multiprocessing.Queue (for separate processes)
            If False, use queue.Queue (for threads)
        """
        if use_multiprocessing:
            self.force_queue = multiprocessing.Queue()
            self.system_queue = multiprocessing.Queue()
            self.response_queue = multiprocessing.Queue()
            self.control_queue = multiprocessing.Queue()  # For control commands
        else:
            self.force_queue = queue.Queue()
            self.system_queue = queue.Queue()
            self.response_queue = queue.Queue()
            self.control_queue = queue.Queue()  # For control commands
    
    def send_force(self, data: ForceData):
        """Send force data from INPUT to OUTPUT"""
        self.force_queue.put(data)
    
    def receive_force(self, timeout: Optional[float] = None) -> Optional[ForceData]:
        """Receive force data in OUTPUT"""
        try:
            return self.force_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def send_system(self, data: SystemData):
        """Send system data from SYSTEM to OUTPUT"""
        self.system_queue.put(data)
    
    def receive_system(self, timeout: Optional[float] = None) -> Optional[SystemData]:
        """Receive system data in OUTPUT"""
        try:
            return self.system_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def send_response(self, data: ResponseData):
        """Send response data from OUTPUT to DASHBOARD"""
        self.response_queue.put(data)
    
    def receive_response(self, timeout: Optional[float] = None) -> Optional[ResponseData]:
        """Receive response data in DASHBOARD"""
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def send_control(self, command: str, data: dict = None):
        """Send control command to INPUT module"""
        self.control_queue.put({'command': command, 'data': data or {}})
    
    def receive_control(self, timeout: Optional[float] = None) -> Optional[dict]:
        """Receive control command in INPUT module"""
        try:
            return self.control_queue.get(timeout=timeout)
        except queue.Empty:
            return None

