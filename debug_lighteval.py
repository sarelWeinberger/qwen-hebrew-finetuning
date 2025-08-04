#!/usr/bin/env python3
"""
LightEval Command Executor with Full Error Capture
"""

import subprocess
import sys
import os

def run_with_full_output(cmd):
    """Run command and capture all output."""
    print(f"🔄 Running: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Read output line by line
        for line in process.stdout:
            print(line, end='')
        
        # Wait for completion
        process.wait()
        
        print(f"\n🏁 Process finished with return code: {process.returncode}")
        return process.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    """Test LightEval commands with full output."""
    
    print("🚀 LightEval Command Tester")
    print("=" * 60)
    
    # Set environment
    env = os.environ.copy()
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['PYTHONPATH'] = os.getcwd()
    
    # Apply environment
    for key, value in env.items():
        os.environ[key] = value
    
    # Commands to test
    test_commands = [
        # Test 1: Simple help
        ['lighteval', '--help'],
        
        # Test 2: Accelerate help
        ['lighteval', 'accelerate', '--help'],
        
        # Test 3: Try the actual command
        ['lighteval', 'accelerate', 'model_name=openai-community/gpt2', 'leaderboard|truthfulqa:mc|0|0'],
        
        # Test 4: Try with output directory
        ['lighteval', 'accelerate', 'model_name=openai-community/gpt2', 'leaderboard|truthfulqa:mc|0|0', '--output-dir', './results'],
        
        # Test 5: Try with different task format
        ['lighteval', 'accelerate', 'model_name=openai-community/gpt2', 'truthfulqa'],
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_commands)}")
        success = run_with_full_output(cmd)
        
        if success:
            print("✅ Command succeeded!")
            break
        else:
            print("❌ Command failed, trying next...")
            if i < len(test_commands):
                print("Continuing to next test...")

if __name__ == "__main__":
    main()
