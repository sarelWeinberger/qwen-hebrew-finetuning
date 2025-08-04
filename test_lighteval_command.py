#!/usr/bin/env python3
"""
Direct LightEval Command Runner

This script runs the exact LightEval command as requested.
"""

import subprocess
import sys
import os

def main():
    """Run LightEval with exact command from user."""
    
    print("🚀 Running LightEval with exact command...")
    print("=" * 60)
    
    # Set environment
    env = os.environ.copy()
    env['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Try different command formats based on LightEval documentation
    commands_to_try = [
        # Original format from user
        ['lighteval', 'accelerate', 'model_name=openai-community/gpt2', 'leaderboard|truthfulqa:mc|0|0'],
        
        # Standard format with flags
        ['lighteval', 'accelerate', 
         '--model_args', 'model_name=openai-community/gpt2',
         '--tasks', 'leaderboard|truthfulqa:mc|0|0',
         '--output_dir', './lighteval_results'],
        
        # Alternative format
        ['lighteval', 'accelerate', 
         '--model_config', 'model_name=openai-community/gpt2',
         '--tasks', 'leaderboard|truthfulqa:mc|0|0'],
        
        # Python module approach
        ['python', '-m', 'lighteval', 'accelerate',
         '--model_args', 'model_name=openai-community/gpt2',
         '--tasks', 'leaderboard|truthfulqa:mc|0|0']
    ]
    
    for i, cmd in enumerate(commands_to_try, 1):
        print(f"\n🔄 Attempt {i}: {' '.join(cmd)}")
        print("-" * 40)
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout for initial attempts
            )
            
            if result.returncode == 0:
                print("✅ Success!")
                print("STDOUT:", result.stdout[:1000] + "..." if len(result.stdout) > 1000 else result.stdout)
                return True
            else:
                print(f"❌ Failed with code {result.returncode}")
                if result.stderr:
                    print("STDERR:", result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
                if result.stdout:
                    print("STDOUT:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
                    
        except subprocess.TimeoutExpired:
            print("⏰ Command timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n💡 Let me try to get help and understand the correct format...")
    
    # Try to get help
    help_commands = [
        ['lighteval', '--help'],
        ['lighteval', 'accelerate', '--help'],
        ['python', '-m', 'lighteval', '--help']
    ]
    
    for cmd in help_commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout:
                print(f"\n📖 Help from: {' '.join(cmd)}")
                print(result.stdout[:800] + "..." if len(result.stdout) > 800 else result.stdout)
                break
        except:
            continue

if __name__ == "__main__":
    main()
