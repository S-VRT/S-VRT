#!/usr/bin/env python3
"""
Test script to verify Flash Attention integration in S-VRT.

This script tests:
1. Numerical parity between flash-attn and original softmax attention
2. Memory usage comparison
3. Performance comparison
"""

import torch
import torch.nn as nn
import numpy as np
import time
import gc
from contextlib import contextmanager

# Import S-VRT components
from models.architectures.vrt.attention import WindowAttention

@contextmanager
def gpu_memory_monitor(stage_name=""):
    """Context manager to monitor GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        yield
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        print(".2f")
    else:
        yield

def test_attention_numerical_parity():
    """Test that flash-attn produces numerically similar results to original attention"""
    print("Testing numerical parity between Flash Attention and original attention...")

    # Test parameters - use smaller window to avoid mutual attention complexity
    batch_size = 1
    num_windows = 1
    window_size = (2, 2, 2)  # Small window: 2*2*2 = 8 tokens
    seq_len = 8
    num_heads = 2  # Small number of heads
    dim = 32  # Small dimension for testing

    # Create test input - single window to avoid mutual attention
    B_ = batch_size * num_windows
    N = seq_len
    C = dim

    # Input tensor: (B_, N, C) - use fp16 for Flash Attention compatibility
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(B_, N, C, dtype=dtype, device=device)
    x_fa = x.clone()  # For flash attention test

    mask = None  # No mask for simplicity

    # Test original attention (use_flash_attn=False, mut_attn=False)
    attn_orig = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads,
                               use_flash_attn=False, mut_attn=False)
    attn_orig = attn_orig.to(dtype=dtype, device=device)
    attn_orig.eval()

    with torch.no_grad():
        out_orig = attn_orig(x)

    # Test flash attention (use_flash_attn=True, mut_attn=False)
    # Use the same module to ensure identical weights
    attn_flash = attn_orig  # Use the same module!
    attn_flash.use_flash_attn = True  # Just change the flag

    with torch.no_grad():
        out_flash = attn_flash(x_fa)

    # Compare outputs
    diff = torch.abs(out_orig - out_flash)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    relative_diff = torch.mean(diff / (torch.abs(out_orig) + 1e-8))

    print(".2e")
    print(".2e")
    print(".2e")

    # Debug: print some statistics
    print(f"Out_orig range: [{out_orig.min().item():.4f}, {out_orig.max().item():.4f}]")
    print(f"Out_flash range: [{out_flash.min().item():.4f}, {out_flash.max().item():.4f}]")
    print(f"Output shapes: orig={out_orig.shape}, flash={out_flash.shape}")
    if max_diff > 0:
        print(f"Max abs difference location: {torch.argmax(diff)}")
        print(f"Sample values - orig[0,0,:3]: {out_orig[0,0,:3].tolist()}")
        print(f"Sample values - flash[0,0,:3]: {out_flash[0,0,:3].tolist()}")

    # Check if results are numerically close
    atol = 1e-4  # Relaxed tolerance for initial testing
    rtol = 1e-2
    is_close = torch.allclose(out_orig, out_flash, atol=atol, rtol=rtol)

    if is_close:
        print(f"✓ PASS: Results are numerically close (atol={atol}, rtol={rtol})")
    else:
        print(f"✗ FAIL: Results differ significantly")
        return False

    return True

def test_attention_performance():
    """Test performance comparison between flash-attn and original attention"""
    print("\nTesting performance comparison...")

    # Test parameters for performance measurement
    batch_size = 2
    num_windows = 4
    window_size = (4, 4, 4)  # 4*4*4 = 64 tokens
    seq_len = 64
    num_heads = 6
    dim = 120

    B_ = batch_size * num_windows
    N = seq_len
    C = dim

    # Use fp16 for Flash Attention compatibility on CUDA, fp32 for CPU
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(B_, N, C, dtype=dtype, device=device)

    # Test original attention (disable mutual attention for fair comparison)
    attn_orig = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads,
                               use_flash_attn=False, mut_attn=False)
    attn_orig = attn_orig.to(dtype=dtype, device=device)
    attn_orig.eval()

    # Test flash attention (disable mutual attention for fair comparison)
    attn_flash = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads,
                                use_flash_attn=True, mut_attn=False)
    attn_flash = attn_flash.to(dtype=dtype, device=device)
    attn_flash.eval()

    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = attn_orig(x.clone())
            _ = attn_flash(x.clone())

    # Measure original attention
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with gpu_memory_monitor("Original Attention"):
        with torch.no_grad():
            for _ in range(10):
                out_orig = attn_orig(x)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    orig_time = (time.time() - start_time) / 10

    # Measure flash attention
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with gpu_memory_monitor("Flash Attention"):
        with torch.no_grad():
            for _ in range(10):
                out_flash = attn_flash(x)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    flash_time = (time.time() - start_time) / 10

    speedup = orig_time / flash_time if flash_time > 0 else float('inf')
    print(".4f")
    print(".4f")
    print(".2f")

    return speedup > 1.0

def test_flash_attn_availability():
    """Test if flash-attn is available and working"""
    print("Testing Flash Attention availability...")

    try:
        from flash_attn import flash_attn_func
        print("✓ Flash Attention library is available")
        return True
    except ImportError as e:
        print(f"✗ Flash Attention library not available: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("S-VRT Flash Attention Integration Test")
    print("=" * 60)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    # Test 1: Availability
    flash_available = test_flash_attn_availability()
    if not flash_available:
        print("Skipping further tests as Flash Attention is not available")
        return

    # Test 2: Numerical parity
    parity_ok = test_attention_numerical_parity()

    # Test 3: Performance
    performance_ok = test_attention_performance()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Flash Attention Available: {'✓' if flash_available else '✗'}")
    print(f"Numerical Parity: {'✓' if parity_ok else '✗'}")
    print(f"Performance Benefit: {'✓' if performance_ok else '✗'}")

    if flash_available and parity_ok and performance_ok:
        print("\n🎉 ALL TESTS PASSED! Flash Attention integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
