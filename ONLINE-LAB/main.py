"""
Main entry point for real-time digital twin system
Starts all modules: INPUT, SYSTEM, OUTPUT, and DASHBOARD
"""
import multiprocessing
import sys
import os
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.communication import CommunicationChannels
from INPUT.force_generator import ForceGenerator
from SYSTEM.system_updater import SystemUpdater
from OUTPUT.response_solver import RealtimeSolver
from DASHBOARD.server import run_dashboard

def run_force_generator(channels_dict):
    """Run force generator in separate process"""
    # Reconstruct channels from dict
    from shared.communication import CommunicationChannels
    channels = CommunicationChannels(use_multiprocessing=True)
    channels.force_queue = channels_dict['force_queue']
    channels.system_queue = channels_dict['system_queue']
    channels.response_queue = channels_dict['response_queue']
    channels.control_queue = channels_dict['control_queue']
    
    # Matching SYSID: use pure Gaussian white noise (no pulses)
    # Reduced noise strength from 100.0 to 10.0 N for better signal quality
    generator = ForceGenerator(channels, use_pulses=False, noise_std=10.0)
    generator.run()

def run_system_updater(channels_dict):
    """Run system updater in separate process"""
    from shared.communication import CommunicationChannels
    channels = CommunicationChannels(use_multiprocessing=True)
    channels.force_queue = channels_dict['force_queue']
    channels.system_queue = channels_dict['system_queue']
    channels.response_queue = channels_dict['response_queue']
    
    updater = SystemUpdater(channels, time_varying=False)
    updater.run()

def run_response_solver(channels_dict):
    """Run response solver in separate process"""
    from shared.communication import CommunicationChannels
    channels = CommunicationChannels(use_multiprocessing=True)
    channels.force_queue = channels_dict['force_queue']
    channels.system_queue = channels_dict['system_queue']
    channels.response_queue = channels_dict['response_queue']
    
    solver = RealtimeSolver(channels)
    solver.run()

def main():
    """Main function to start all modules"""
    print("=" * 60)
    print("Real-time Digital Twin System - 5DOF Model")
    print("=" * 60)
    
    # Create communication channels with multiprocessing support
    channels = CommunicationChannels(use_multiprocessing=True)
    
    # Create channels dict for passing to processes
    channels_dict = {
        'force_queue': channels.force_queue,
        'system_queue': channels.system_queue,
        'response_queue': channels.response_queue,
        'control_queue': channels.control_queue
    }
    
    # Create a separate channels object for dashboard control (shares control_queue)
    dashboard_control_channels = CommunicationChannels(use_multiprocessing=True)
    dashboard_control_channels.control_queue = channels.control_queue
    
    # Create processes
    processes = []
    
    # Start INPUT process
    print("Starting INPUT module (Force Generator)...")
    p_input = multiprocessing.Process(target=run_force_generator, args=(channels_dict,))
    p_input.start()
    processes.append(p_input)
    time.sleep(0.5)
    
    # Start SYSTEM process
    print("Starting SYSTEM module (System Updater)...")
    p_system = multiprocessing.Process(target=run_system_updater, args=(channels_dict,))
    p_system.start()
    processes.append(p_system)
    time.sleep(0.5)
    
    # Start OUTPUT process
    print("Starting OUTPUT module (Response Solver)...")
    p_output = multiprocessing.Process(target=run_response_solver, args=(channels_dict,))
    p_output.start()
    processes.append(p_output)
    time.sleep(0.5)
    
    # Start DASHBOARD (runs in main process)
    print("Starting DASHBOARD module (Web Server)...")
    print("Dashboard will be available at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop all modules")
    print("=" * 60)
    
    try:
        run_dashboard(channels, dashboard_control_channels)
    except KeyboardInterrupt:
        print("\nShutting down...")
        for p in processes:
            p.terminate()
            p.join()
        print("All modules stopped.")

if __name__ == "__main__":
    main()

