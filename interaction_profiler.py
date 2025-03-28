import cProfile
import pstats
import io
import time
import logging
import functools
import atexit
from datetime import datetime
import contextvars
from fastapi import Request
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a context variable to store the current interaction ID
current_interaction_id = contextvars.ContextVar('current_interaction_id', default=None)

# Store active sessions
class SessionStore:
    def __init__(self):
        self.sessions = {}
        self.client_sessions = {}
        self.session_timeout = 30  # seconds
    
    def get_or_create_session(self, client_id):
        current_time = time.time()
        
        # Check if client has an active session
        if client_id in self.client_sessions:
            session_id = self.client_sessions[client_id]
            session = self.sessions.get(session_id)
            
            # If session exists and is not expired
            if session and (current_time - session['last_activity']) < self.session_timeout:
                # Update last activity
                session['last_activity'] = current_time
                return session_id
        
        # Create new session
        session_id = f"session_{int(current_time * 1000)}"
        self.sessions[session_id] = {
            'start_time': current_time,
            'last_activity': current_time,
            'client_id': client_id,
            'endpoints': [],
            'profile': cProfile.Profile(),
            'phases': [],
            'current_endpoint': None
        }
        
        # Associate with client
        self.client_sessions[client_id] = session_id
        
        # Start profiling
        self.sessions[session_id]['profile'].enable()
        logger.info(f"Created new session {session_id} for client {client_id}")
        
        return session_id
    
    def cleanup_expired_sessions(self):
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if (current_time - session['last_activity']) > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.end_session(session_id)
    
    def mark_phase(self, session_id, phase_name):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            current_time = time.time()
            phase_data = {
                "name": phase_name,
                "timestamp": current_time,
                "endpoint": session['current_endpoint'],
                "elapsed_from_start": current_time - session['start_time']
            }
            session['phases'].append(phase_data)
            session['last_activity'] = current_time
            logger.debug(f"Phase '{phase_name}' marked for session {session_id}")
    
    def register_endpoint_call(self, session_id, endpoint_name):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session['endpoints'].append(endpoint_name)
            session['current_endpoint'] = endpoint_name
            session['last_activity'] = time.time()
            logger.debug(f"Registered endpoint {endpoint_name} for session {session_id}")
    
    def end_session(self, session_id):
        if session_id in self.sessions:
            try:
                session = self.sessions[session_id]
                profile = session['profile']
                profile.disable()
                
                # Remove client association
                if session['client_id'] in self.client_sessions:
                    if self.client_sessions[session['client_id']] == session_id:
                        del self.client_sessions[session['client_id']]
                
                # Generate reports
                output_dir = "./profiles/sessions"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save profiling data
                filename = f"{output_dir}/{session_id}_{timestamp}.prof"
                with open(filename, "w") as f:
                    ps = pstats.Stats(profile, stream=f)
                    ps.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(30)
                
                # Save timing data
                timing_file = f"{output_dir}/{session_id}_{timestamp}.txt"
                with open(timing_file, "w") as f:
                    total_time = time.time() - session['start_time']
                    
                    f.write(f"Session {session_id} Total Time: {total_time:.4f} seconds\n\n")
                    f.write(f"Endpoints Called: {', '.join(session['endpoints'])}\n\n")
                    f.write("Phase Breakdown:\n")
                    
                    last_time = session['start_time']
                    for phase in session['phases']:
                        phase_time = phase["timestamp"]
                        since_last = phase_time - last_time
                        f.write(f"  {phase['endpoint']} - {phase['name']}: {since_last:.4f}s (elapsed: {phase['elapsed_from_start']:.4f}s)\n")
                        last_time = phase_time
                    
                    f.write(f"\nTotal interaction time: {total_time:.4f}s\n")
                
                logger.info(f"Session {session_id} saved to {filename}")
                
                # Clean up
                del self.sessions[session_id]
                
            except Exception as e:
                logger.error(f"Error ending session {session_id}: {str(e)}")
                
                # Clean up even if there was an error
                if session_id in self.sessions:
                    del self.sessions[session_id]

# Create session store
session_store = SessionStore()

# Run cleanup periodically
def cleanup_task():
    while True:
        time.sleep(60)  # Check every minute
        session_store.cleanup_expired_sessions()

# Start cleanup in a background thread
import threading
cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
cleanup_thread.start()

# Function to get client identifier from request
def get_client_id(request):
    # Use client IP + User-Agent as identifier
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    return f"{client_ip}_{hash(user_agent)}"

# Decorator for profiling with session tracking
def profile_session(endpoint_name=None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract the request object
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                for key, value in kwargs.items():
                    if isinstance(value, Request):
                        request = value
                        break
            
            if not request:
                # Fall back to regular profiling if no request object
                return await func(*args, **kwargs)
            
            # Get client identifier and session
            client_id = get_client_id(request)
            session_id = session_store.get_or_create_session(client_id)
            
            # Register this endpoint call
            ep_name = endpoint_name or func.__name__
            session_store.register_endpoint_call(session_id, ep_name)
            
            # Set context for mark_phase function
            token = current_interaction_id.set(session_id)
            
            try:
                # Mark start of processing
                session_store.mark_phase(session_id, "start_processing")
                
                # Call the original function
                result = await func(*args, **kwargs)
                
                # Mark end of processing
                session_store.mark_phase(session_id, "end_processing")
                
                return result
            except Exception as e:
                session_store.mark_phase(session_id, f"error: {str(e)}")
                raise
            finally:
                current_interaction_id.reset(token)
            
        return wrapper
    return decorator

# Simple function to mark phases in the current session
def mark_session_phase(phase_name):
    session_id = current_interaction_id.get()
    if session_id:
        session_store.mark_phase(session_id, phase_name)
    else:
        logger.warning(f"Cannot mark phase '{phase_name}': No active session")

# Function to end the current session (for manual control)
def end_current_session():
    session_id = current_interaction_id.get()
    if session_id:
        session_store.end_session(session_id)
        return True
    return False

# Register end_all_sessions to run at exit
@atexit.register
def end_all_sessions():
    for session_id in list(session_store.sessions.keys()):
        session_store.end_session(session_id) 