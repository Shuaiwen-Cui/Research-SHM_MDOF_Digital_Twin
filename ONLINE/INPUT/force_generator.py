"""
Force generator: generates white noise + periodic pulse loads
"""
import numpy as np
import time
import sys
import os

# Add ONLINE directory to path
ONLINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ONLINE_DIR not in sys.path:
    sys.path.insert(0, ONLINE_DIR)

from shared.data_protocol import ForceData
from shared.communication import CommunicationChannels
from config.system_config import nDOF, dt, m_initial

# Import earthquake and strongwind modules from same directory
try:
    from .earthquake import earthquake_gen
    from .strongwind import strong_wind_gen
except ImportError:
    # Fallback for direct execution
    import earthquake
    import strongwind
    earthquake_gen = earthquake.earthquake_gen
    strong_wind_gen = strongwind.strong_wind_gen

class ForceGenerator:
    """Generates force vectors with white noise + periodic pulses"""
    
    def __init__(self, channels: CommunicationChannels, 
                 noise_std: float = 100.0,
                 pulse_interval: float = 10.0,
                 pulse_duration: float = 0.1,
                 pulse_amplitude: float = 5000.0):
        """
        Parameters:
        -----------
        channels: CommunicationChannels
            Communication channels for sending data
        noise_std: float
            Standard deviation of white noise (N)
        pulse_interval: float
            Time interval between pulses (seconds)
        pulse_duration: float
            Duration of each pulse (seconds)
        pulse_amplitude: float
            Amplitude of pulse load (N)
        """
        self.channels = channels
        self.noise_std = noise_std
        self.pulse_interval = pulse_interval
        self.pulse_duration = pulse_duration
        self.pulse_amplitude = pulse_amplitude
        
        self.current_time = 0.0
        self.next_pulse_time = self.pulse_interval
        self.pulse_active = False
        self.pulse_end_time = 0.0
        self.pulse_dof = 0  # DOF where pulse is applied
        
        # Mode control
        self.auto_mode = True  # True for auto, False for manual
        self.manual_trigger = False  # Flag for manual impact trigger
        self.current_pulse_amplitude = pulse_amplitude  # Current pulse amplitude
        
        # Event buffers for earthquake and strong wind
        self.earthquake_buffer = None  # Pre-generated earthquake acceleration time history
        self.earthquake_start_time = None
        self.earthquake_active = False
        
        self.strongwind_buffer = None  # Pre-generated strong wind load time history (nDOF x signal_length)
        self.strongwind_start_time = None
        self.strongwind_active = False
        
        # Event parameters
        # Calculate signal length for ~30 seconds of event duration
        self.event_signal_length = int(30.0 / dt)  # ~30 seconds duration
        
    def trigger_manual_impact(self):
        """Trigger a manual impact event"""
        self.manual_trigger = True
    
    def trigger_earthquake(self):
        """Trigger an earthquake event - starts immediately from first point"""
        # Generate earthquake ground acceleration
        eq_strength = 0.6
        eq_duration = 20.0
        omegag = 15
        zetag = 0.6
        fac_time = 12.21
        pos_time1 = 0.1
        pos_time2 = 0.5
        
        # For real-time system: generate earthquake that starts immediately
        # Calculate actual earthquake length based on duration
        eq_length = int(eq_duration / dt)
        
        # Generate earthquake using the standard function
        # But we need to extract the actual earthquake wave (Ag) directly
        # instead of using the random start_point logic
        try:
            from .earthquake import spec_density
        except ImportError:
            from earthquake import spec_density
        
        # Generate the actual earthquake wave (similar to earthquake_gen but without random start)
        num_terms = 200
        vl = -np.pi
        vu = np.pi
        
        # Time vector for earthquake duration
        t_vec = np.arange(dt, eq_duration + dt, dt)
        
        # Frequency parameters
        wu = 1 / dt / 2
        dw = wu / num_terms
        k = np.arange(1, num_terms + 1)
        
        # Time modulating function
        gt = fac_time * (np.exp(-pos_time1 * t_vec) - np.exp(-pos_time2 * t_vec))
        
        # Random variables
        theta = vl + (vu - vl) * np.random.randn(1)
        Xk = np.zeros((1, num_terms))
        Yk = np.zeros((1, num_terms))
        
        for k_idx in range(num_terms):
            Xk[0, k_idx] = np.sqrt(2) * np.cos(k_idx * theta + np.pi / 4)
            Yk[0, k_idx] = np.sqrt(2) * np.sin(k_idx * theta + np.pi / 4)
        
        Xk = np.apply_along_axis(np.random.permutation, 1, Xk)
        Yk = np.apply_along_axis(np.random.permutation, 1, Yk)
        
        # Generate ground acceleration (starts immediately from first point)
        Ag = np.zeros(len(t_vec))
        for i in range(len(t_vec)):
            Ag_terms = (np.sqrt(2 * spec_density(dw * k, omegag, zetag, eq_strength) * dw) *
                       (np.cos(dw * k * i * dt) * Xk + np.sin(dw * k * i * dt) * Yk))
            Ag_series = np.sum(Ag_terms)
            Ag[i] = Ag_series * gt[i]
        
        # Normalize and rescale (same as original function)
        PGA = max(abs(Ag))
        if PGA > 0:
            rescale_factor = np.random.rand(1) * 0.9 * 9.81 + 0.1 * 9.81
            earthquake_acc = Ag / PGA * rescale_factor
        else:
            earthquake_acc = Ag
        
        self.earthquake_buffer = earthquake_acc
        self.earthquake_start_time = self.current_time
        self.earthquake_active = True
        print(f"Earthquake event triggered, duration: {len(earthquake_acc)*dt:.2f}s (starts immediately)")
    
    def trigger_strongwind(self):
        """Trigger a strong wind event"""
        # Generate strong wind load
        strongwind_load = strong_wind_gen(self.event_signal_length, nDOF, dt)
        
        # Apply scaling factor to reduce wind load magnitude
        # Original values are too large, scale down by factor of 0.001 (reduce by 1000x)
        wind_scale_factor = 0.001
        strongwind_load = strongwind_load * wind_scale_factor
        
        self.strongwind_buffer = strongwind_load
        self.strongwind_start_time = self.current_time
        self.strongwind_active = True
        print(f"Strong wind event triggered, duration: {len(strongwind_load[0])*dt:.2f}s (scaled by {wind_scale_factor})")
    
    def set_mode(self, auto_mode: bool):
        """Set operation mode: True for auto, False for manual"""
        self.auto_mode = auto_mode
        if auto_mode:
            # Reset auto mode timing
            self.next_pulse_time = self.current_time + self.pulse_interval
    
    def generate_force(self) -> np.ndarray:
        """Generate force vector for current time step - all events can be superimposed"""
        force = np.zeros(nDOF)
        
        # Always add white noise (can be superimposed with other events)
        force += np.random.normal(0, self.noise_std, nDOF)
        
        # Handle earthquake event (can be active simultaneously with other events)
        if self.earthquake_active and self.earthquake_buffer is not None:
            time_idx = int((self.current_time - self.earthquake_start_time) / dt)
            if 0 <= time_idx < len(self.earthquake_buffer):
                # Convert ground acceleration to force: F = -M * a_g
                # Earthquake affects all DOFs through mass matrix
                ground_acc = self.earthquake_buffer[time_idx]
                # Apply earthquake force: F = -M @ (ones(nDOF) * ground_acc)
                # For diagonal mass matrix: F_i = -m_i * ground_acc
                force -= m_initial * ground_acc  # Negative because it's ground acceleration
            else:
                # Earthquake event finished
                self.earthquake_active = False
                self.earthquake_buffer = None
                print("Earthquake event finished")
        
        # Handle strong wind event (can be active simultaneously with other events)
        if self.strongwind_active and self.strongwind_buffer is not None:
            time_idx = int((self.current_time - self.strongwind_start_time) / dt)
            if 0 <= time_idx < self.strongwind_buffer.shape[1]:
                # Strong wind load is already in force units (N) for each DOF
                force += self.strongwind_buffer[:, time_idx]
            else:
                # Strong wind event finished
                self.strongwind_active = False
                self.strongwind_buffer = None
                print("Strong wind event finished")
        
        # Handle impact events (can be triggered at any time, including during other events)
        if self.auto_mode:
            # Auto mode: periodic pulses
            if self.current_time >= self.next_pulse_time:
                # Start new pulse
                self.pulse_active = True
                self.pulse_dof = np.random.randint(0, nDOF)  # Random DOF
                self.pulse_end_time = self.current_time + self.pulse_duration
                self.next_pulse_time = self.current_time + self.pulse_interval
                self.current_pulse_amplitude = self.pulse_amplitude
        else:
            # Manual mode: check for manual trigger
            if self.manual_trigger:
                # Start manual impact (can be triggered even during earthquake/wind)
                self.pulse_active = True
                self.pulse_dof = np.random.randint(0, nDOF)  # Random DOF
                # Random amplitude between 0.5x and 1.5x of default
                impact_amplitude = self.pulse_amplitude * (0.5 + np.random.rand())
                self.pulse_end_time = self.current_time + self.pulse_duration
                self.manual_trigger = False
                # Store the amplitude for this pulse
                self.current_pulse_amplitude = impact_amplitude
                print(f"Manual impact triggered during active events (DOF: {self.pulse_dof})")
            elif not self.pulse_active:
                # Reset pulse amplitude to default if no active pulse
                self.current_pulse_amplitude = self.pulse_amplitude
        
        # Apply pulse if active (can be superimposed with earthquake/wind)
        if self.pulse_active and self.current_time < self.pulse_end_time:
            force[self.pulse_dof] += self.current_pulse_amplitude
        elif self.pulse_active and self.current_time >= self.pulse_end_time:
            self.pulse_active = False
        
        return force
    
    def run(self):
        """Main loop: generate and send forces"""
        print("Force Generator started")
        
        while True:
            # Check for control commands
            control_cmd = self.channels.receive_control(timeout=0.001)
            if control_cmd:
                cmd = control_cmd.get('command', '')
                cmd_data = control_cmd.get('data', {})
                
                if cmd == 'set_mode':
                    auto_mode = cmd_data.get('auto_mode', True)
                    self.set_mode(auto_mode)
                    print(f"Mode switched to: {'Auto' if auto_mode else 'Manual'}")
                elif cmd == 'trigger_impact':
                    if not self.auto_mode:
                        self.trigger_manual_impact()
                        print("Manual impact triggered")
                elif cmd == 'trigger_earthquake':
                    if not self.auto_mode:
                        self.trigger_earthquake()
                elif cmd == 'trigger_strongwind':
                    if not self.auto_mode:
                        self.trigger_strongwind()
            
            # Generate force
            force = self.generate_force()
            
            # Create and send data
            force_data = ForceData(
                timestamp=self.current_time,
                force=force
            )
            self.channels.send_force(force_data)
            
            # Update time
            self.current_time += dt
            
            # Sleep to maintain real-time (optional, for simulation)
            time.sleep(dt)

if __name__ == "__main__":
    channels = CommunicationChannels()
    generator = ForceGenerator(channels)
    generator.run()

