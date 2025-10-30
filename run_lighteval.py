#!/usr/bin/env python3
"""
LightEval Runner Script

This script runs LightEval evaluations with proper environment setup.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_lighteval_command():
    """Run the specific LightEval command requested."""
    
    # Set environment variables for better performance
    env = os.environ.copy()
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # The command from the user request
    command = [
        'lighteval', 'accelerate',
        'model_name=openai-community/gpt2',
        'leaderboard|truthfulqa:mc|0|0'
    ]
    
    print("Running LightEval with the following command:")
    print(' '.join(command))
    print("-" * 60)
    
    try:
        # Run the command
        result = subprocess.run(
            command,
            env=env,
            cwd=os.getcwd(),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✅ LightEval completed successfully!")
        else:
            print(f"\n❌ LightEval failed with return code: {result.returncode}")
            
    except FileNotFoundError:
        print("❌ Error: lighteval command not found. Make sure it's installed and in PATH.")
        return False
    except Exception as e:
        print(f"❌ Error running lighteval: {e}")
        return False
    
    return True

def check_lighteval_info():
    """Check available models and tasks in LightEval."""
    
    print("🔍 Checking LightEval installation and available options...")
    print("-" * 60)
    
    try:
        # Check LightEval help
        result = subprocess.run(['lighteval', '--help'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ LightEval is properly installed")
            print("\nAvailable commands:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print("❌ Issue with LightEval installation")
            
    except Exception as e:
        print(f"❌ Error checking LightEval: {e}")

def main():
    """Main function to run LightEval."""
    
    print("🚀 LightEval Runner")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not in virtual environment - this might cause issues")
    
    # Check LightEval installation
    check_lighteval_info()
    
    print("\n" + "="*60)
    
    # Run the actual command
    success = run_lighteval_command()
    
    if success:
        print("\n🎉 Evaluation completed! Check the output above for results.")
    else:
        print("\n💔 Evaluation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
