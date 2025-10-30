import socket
import sys
import time
import subprocess

def is_port_in_use(port, host='0.0.0.0'):
    """Check if a port is currently in use - IMPROVED VERSION"""
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # DON'T use SO_REUSEADDR for checking - it gives false negatives!
        sock.settimeout(1)
        
        # Try to bind - if it fails, port is in use
        sock.bind((host, port))
        
        # Port is free
        return False
        
    except OSError as e:
        # Port is in use
        return True
        
    finally:
        if sock:
            sock.close()

def get_process_on_port(port):
    """Get the process ID using the port"""
    try:
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except:
        return None

def kill_process_on_port(port):
    """Attempt to kill the process using the port"""
    try:
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        pid = result.stdout.strip()
        
        if pid:
            print(f"   Found process {pid} on port {port}")
            subprocess.run(['kill', '-9', pid], timeout=5)
            print(f"   Killed process {pid}")
            time.sleep(2)  # Wait for OS to release the port
            return True
        return False
    except Exception as e:
        print(f"   Failed to kill process: {e}")
        return False

def check_port_or_raise(port, host='0.0.0.0', timeout=5, retry=True, auto_kill=False):
    """
    Check if port is available, with optional retry logic.
    
    Args:
        port (int): Port to check
        host (str): Host address
        timeout (int): Seconds to wait between retries
        retry (bool): Whether to retry if port is busy
        auto_kill (bool): Whether to automatically kill process on port
    
    Raises:
        RuntimeError: If port is unavailable after all retries
    """
    print(f"Checking port {port}...")
    
    if not is_port_in_use(port, host):
        print(f"✓ Port {port} is available")
        return True
    
    # Port is in use
    pid = get_process_on_port(port)
    
    print(f"⚠ Port {port} is busy")
    if pid:
        print(f"   Process ID: {pid}")
    
    # Try to auto-kill if requested
    if auto_kill:
        print(f"   Attempting to kill process on port {port}...")
        if kill_process_on_port(port):
            # Check if port is now free
            if not is_port_in_use(port, host):
                print(f"✓ Port {port} is now available")
                return True
    
    # Retry logic
    if retry:
        print(f"   Waiting {timeout} seconds for port to be released...")
        time.sleep(timeout)
        
        # Check again
        if not is_port_in_use(port, host):
            print(f"✓ Port {port} is now available")
            return True
    
    # Still not available - raise error
    error_msg = f"❌ Port {port} is NOT available!"
    if pid:
        error_msg += f"\n   Process ID: {pid}"
    error_msg += f"\n   To kill manually: lsof -ti:{port} | xargs kill -9"
    
    raise RuntimeError(error_msg)

# Usage example
if __name__ == "__main__":
    PORT = 7680
    
    try:
        # Option 1: Just check (no auto-kill)
        # check_port_or_raise(PORT, retry=False, auto_kill=False)
        
        # Option 2: Auto-kill the process if found
        check_port_or_raise(PORT, retry=True, auto_kill=True, timeout=3)
        
        print(f"🚀 You can start application on port {PORT}...")
        
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)