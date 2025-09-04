#!/usr/bin/env python3
"""
CUDA initialization script to fix post-reboot CUDA issues
"""
import os
import sys

# Set CUDA device order to PCI_BUS_ID for consistent device mapping
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA compiled version:", torch.version.cuda)
    
    # Force CUDA initialization
    if hasattr(torch.cuda, '_lazy_init'):
        torch.cuda._lazy_init()
    
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, Memory: {props.total_memory/1024**3:.1f}GB")
        
        # Test a simple CUDA operation
        device = torch.device("cuda:0")
        x = torch.tensor([1.0], device=device)
        y = x + 1
        print("Simple CUDA operation successful:", y.item())
        print("✅ CUDA is working properly!")
    else:
        print("❌ CUDA is not available")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
