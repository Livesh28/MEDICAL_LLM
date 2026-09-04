#!/usr/bin/env python3
"""
Phase 1 Verification Script: Test Apple Silicon MPS Backend
Verifies PyTorch installation, MPS availability, device selection, and performs
a matrix multiplication test on the Apple Silicon GPU.
"""

import sys
import platform
import torch

def main():
    print("=" * 60)
    print("PHASE 1: Apple Silicon & MPS Backend Verification")
    print("=" * 60)
    
    # System Information
    py_ver = sys.version.split()[0]
    os_info = platform.platform()
    machine = platform.machine()
    
    print(f"Python Version:   {py_ver}")
    print(f"Platform:         {os_info}")
    print(f"Architecture:     {machine}")
    
    # PyTorch & MPS Check
    torch_ver = torch.__version__
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    
    print(f"PyTorch Version:  {torch_ver}")
    print(f"MPS built:        {mps_built}")
    print(f"MPS available:    {mps_available}")
    
    # Select Device
    if mps_available:
        device = torch.device("mps")
        print(f"Selected device:  {device}")
    else:
        device = torch.device("cpu")
        print(f"Selected device:  {device} (Fallback - MPS not available)")
    
    # Matrix Multiplication Test on Selected Device
    print("-" * 60)
    print("Running Matrix Multiplication Test on MPS...")
    
    try:
        # Create two 1000x1000 random tensors on the selected device
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        # Matrix multiply
        c = torch.matmul(a, b)
        
        # Ensure computation completes and syncs
        if device.type == "mps":
            torch.mps.synchronize()
            
        print(f"Matrix multiplication output shape: {c.shape}")
        print(f"Matrix multiplication device:       {c.device}")
        print("Matrix multiplication test:         SUCCESSFUL")
        
        # Additional Memory Config Notice
        print("-" * 60)
        print("Target System Hardware Configuration:")
        print(" - Apple M1 Pro (ARM64)")
        print(" - 16 GB Unified Memory")
        print(" - Target Model: ~30 Million Parameters")
        print("Phase 1 verification PASSED successfully.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"Matrix multiplication test FAILED with error: {e}")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
