#!/usr/bin/env python3
"""
Simple LightEval test to verify setup works
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, '.')

try:
    # Try importing key components
    from lighteval.main_accelerate import main as accelerate_main
    from lighteval.utils.utils import obj_to_pydantic_config
    
    print("✅ LightEval imports successful")
    
    # Try to see what's available
    try:
        from lighteval.tasks.registry import Registry
        registry = Registry()
        task_list = registry.get_task_list()
        print(f"✅ Found {len(task_list)} available tasks")
        
        # Show some tasks
        print("\nSome available tasks:")
        for task in sorted(task_list)[:10]:
            print(f"  - {task}")
        
        # Check for truthfulqa specifically
        truthful_tasks = [t for t in task_list if 'truthful' in t.lower()]
        if truthful_tasks:
            print(f"\nTruthfulQA tasks found:")
            for task in truthful_tasks:
                print(f"  - {task}")
        else:
            print("\n❌ No TruthfulQA tasks found")
            
    except Exception as e:
        print(f"❌ Error loading task registry: {e}")
        
    # Try a simple command structure
    print("\n" + "="*50)
    print("Testing command structure...")
    
    # Set up basic args for testing
    model_args = "model_name=openai-community/gpt2"
    
    print(f"Model args: {model_args}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
