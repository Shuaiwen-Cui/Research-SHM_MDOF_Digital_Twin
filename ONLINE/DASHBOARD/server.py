"""
Web server for real-time monitoring dashboard
"""
import sys
import os
import threading
import time

# Add ONLINE directory to path
ONLINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ONLINE_DIR not in sys.path:
    sys.path.insert(0, ONLINE_DIR)

from shared.communication import CommunicationChannels
from shared.data_protocol import ResponseData

try:
    from flask import Flask, render_template
    from flask_socketio import SocketIO, emit
except ImportError:
    print("Flask and flask-socketio are required. Install with: pip install flask flask-socketio")
    sys.exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'digital_twin_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global communication channels (will be set by main)
channels = None
force_generator_channels = None  # For sending control commands to INPUT

def data_collector():
    """Background thread to collect response data and send to clients"""
    global channels
    if channels is None:
        return
    
    while True:
        response_data = channels.receive_response(timeout=0.1)
        if response_data is not None:
            # Send to all connected clients via WebSocket
            socketio.emit('response_data', {
                'timestamp': response_data.timestamp,
                'displacement': response_data.displacement.tolist(),
                'velocity': response_data.velocity.tolist(),
                'acceleration': response_data.acceleration.tolist()
            })

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'data': 'Connected to Digital Twin Dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('set_mode')
def handle_set_mode(data):
    """Handle mode switch command"""
    global force_generator_channels
    if force_generator_channels:
        auto_mode = data.get('auto_mode', True)
        force_generator_channels.send_control('set_mode', {'auto_mode': auto_mode})
        print(f"Mode switch command: {'Auto' if auto_mode else 'Manual'}")

@socketio.on('trigger_impact')
def handle_trigger_impact():
    """Handle manual impact trigger"""
    global force_generator_channels
    if force_generator_channels:
        force_generator_channels.send_control('trigger_impact', {})
        print("Manual impact trigger command sent")

@socketio.on('trigger_earthquake')
def handle_trigger_earthquake():
    """Handle earthquake trigger"""
    global force_generator_channels
    if force_generator_channels:
        force_generator_channels.send_control('trigger_earthquake', {})
        print("Earthquake trigger command sent")

@socketio.on('trigger_strongwind')
def handle_trigger_strongwind():
    """Handle strong wind trigger"""
    global force_generator_channels
    if force_generator_channels:
        force_generator_channels.send_control('trigger_strongwind', {})
        print("Strong wind trigger command sent")

def run_dashboard(comm_channels: CommunicationChannels, force_gen_channels: CommunicationChannels = None, host='127.0.0.1', port=5000):
    """Run the dashboard server"""
    global channels, force_generator_channels
    channels = comm_channels
    force_generator_channels = force_gen_channels
    
    # Start data collector thread
    collector_thread = threading.Thread(target=data_collector, daemon=True)
    collector_thread.start()
    
    print(f"Dashboard server starting on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    # For testing
    channels = CommunicationChannels()
    run_dashboard(channels)

