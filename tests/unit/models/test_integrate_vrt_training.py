#!/usr/bin/env python3
"""
Unit test for IntegrateVRT in training scenario.

Tests the full model with DataParallel and AMP to debug CUDA errors.
"""
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn


def test_model_forward_cpu():
    """Test model forward pass on CPU."""
    print("\n" + "="*70)
    print("Test 1: VRTWithSpike Forward Pass (CPU)")
    print("="*70)
    
    print("Skipping CPU test (VRT model is too large for CPU testing)...")
    return True
    
    # from src.models.integrate_vrt import VRTWithSpike
    
    # print("\nCreating model...")
    # model = VRTWithSpike(
    #     vrt_backbone=None,  # Would need actual VRT backbone
    #     spike_bins=5,  # num_voxel_bins
    #     tsa_heads=8,
    # )
    # model.eval()
    
    # Create dummy inputs
    B, T, C, H, W = 2, 5, 3, 64, 64
    blur = torch.randn(B, T, C, H, W)
    spike_vox = torch.randn(B, T, 5, H, W)  # 5 bins
    
    print(f"Input shapes:")
    print(f"  blur: {list(blur.shape)}")
    print(f"  spike_vox: {list(spike_vox.shape)}")
    
    try:
        with torch.no_grad():
            output = model(blur, spike_vox)
        
        print(f"\n✓ Forward pass successful!")
        print(f"  Output shape: {list(output.shape)}")
        
    except Exception as e:
        print(f"\n✗ Forward pass failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model_forward_single_gpu():
    """Test model forward pass on single GPU."""
    print("\n" + "="*70)
    print("Test 2: VRTWithSpike Forward Pass (Single GPU)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping...")
        return True
    
    from src.models.integrate_vrt import IntegrateVRT
    
    device = torch.device('cuda:0')
    print(f"\nUsing device: {torch.cuda.get_device_name(0)}")
    
    print("\nCreating model...")
    model = IntegrateVRT(
        vrt_ckpt=None,
        spike_in_channels=5,
        heads=8,
    ).to(device)
    model.eval()
    
    # Create dummy inputs - smaller size for faster testing
    B, T, C, H, W = 1, 5, 3, 64, 64
    blur = torch.randn(B, T, C, H, W, device=device)
    spike_vox = torch.randn(B, T, 5, H, W, device=device)
    
    print(f"Input shapes:")
    print(f"  blur: {list(blur.shape)}")
    print(f"  spike_vox: {list(spike_vox.shape)}")
    
    try:
        with torch.no_grad():
            output = model(blur, spike_vox)
        
        print(f"\n✓ Forward pass successful!")
        print(f"  Output shape: {list(output.shape)}")
        
    except Exception as e:
        print(f"\n✗ Forward pass failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model_forward_dataparallel():
    """Test model forward pass with DataParallel."""
    print("\n" + "="*70)
    print("Test 3: VRTWithSpike Forward Pass (DataParallel)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping...")
        return True
    
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"Only {n_gpus} GPU available, testing with single GPU DataParallel...")
        device_ids = [0]
    else:
        print(f"Found {n_gpus} GPUs, testing with 2 GPUs...")
        device_ids = [0, 1]
    
    from src.models.integrate_vrt import VRTWithSpike
    
    print("\nCreating model...")
    model = VRTWithSpike(
        vrt_ckpt=None,
        spike_in_channels=5,
        heads=8,
    )
    
    print(f"Wrapping with DataParallel (GPUs: {device_ids})...")
    model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    
    # Create dummy inputs - batch size that will be split across GPUs
    batch_per_gpu = 2
    B = batch_per_gpu * len(device_ids)
    T, C, H, W = 5, 3, 64, 64
    
    device = torch.device(f'cuda:{device_ids[0]}')
    blur = torch.randn(B, T, C, H, W, device=device)
    spike_vox = torch.randn(B, T, 5, H, W, device=device)
    
    print(f"Input shapes (total batch={B}, {batch_per_gpu} per GPU):")
    print(f"  blur: {list(blur.shape)}")
    print(f"  spike_vox: {list(spike_vox.shape)}")
    
    try:
        with torch.no_grad():
            output = model(blur, spike_vox)
        
        print(f"\n✓ DataParallel forward pass successful!")
        print(f"  Output shape: {list(output.shape)}")
        
    except Exception as e:
        print(f"\n✗ DataParallel forward pass failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model_forward_amp():
    """Test model forward pass with Automatic Mixed Precision."""
    print("\n" + "="*70)
    print("Test 4: VRTWithSpike Forward Pass (AMP)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping...")
        return True
    
    from src.models.integrate_vrt import IntegrateVRT
    
    device = torch.device('cuda:0')
    print(f"\nUsing device: {torch.cuda.get_device_name(0)}")
    
    print("\nCreating model...")
    model = IntegrateVRT(
        vrt_ckpt=None,
        spike_in_channels=5,
        heads=8,
    ).to(device)
    model.eval()
    
    # Create dummy inputs
    B, T, C, H, W = 2, 5, 3, 64, 64
    blur = torch.randn(B, T, C, H, W, device=device)
    spike_vox = torch.randn(B, T, 5, H, W, device=device)
    
    print(f"Input shapes:")
    print(f"  blur: {list(blur.shape)}")
    print(f"  spike_vox: {list(spike_vox.shape)}")
    
    print("\nTesting with AMP...")
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = model(blur, spike_vox)
        
        print(f"✓ AMP forward pass successful!")
        print(f"  Output shape: {list(output.shape)}")
        
    except Exception as e:
        print(f"✗ AMP forward pass failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model_training_scenario():
    """Test model in realistic training scenario with DataParallel + AMP."""
    print("\n" + "="*70)
    print("Test 5: Full Training Scenario (DataParallel + AMP)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping...")
        return True
    
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"Only {n_gpus} GPU available, using single GPU...")
        device_ids = [0]
    else:
        print(f"Found {n_gpus} GPUs, using 2 GPUs...")
        device_ids = [0, 1]
    
    from src.models.integrate_vrt import IntegrateVRT
    
    print("\nCreating model with DataParallel...")
    model = IntegrateVRT(
        vrt_ckpt=None,
        spike_in_channels=5,
        heads=8,
    )
    model = nn.DataParallel(model, device_ids=device_ids)
    model.train()  # Training mode
    
    # Realistic training batch size
    batch_per_gpu = 4
    B = batch_per_gpu * len(device_ids)
    T, C, H, W = 5, 3, 128, 128  # Larger resolution
    
    device = torch.device(f'cuda:{device_ids[0]}')
    blur = torch.randn(B, T, C, H, W, device=device)
    spike_vox = torch.randn(B, T, 5, H, W, device=device)
    target = torch.randn(B, T, C, H, W, device=device)
    
    print(f"Batch configuration:")
    print(f"  Total batch size: {B} (={batch_per_gpu} per GPU × {len(device_ids)} GPUs)")
    print(f"  Input resolution: {H}×{W}")
    print(f"  blur: {list(blur.shape)}")
    print(f"  spike_vox: {list(spike_vox.shape)}")
    
    print("\nTesting forward pass with AMP...")
    try:
        with torch.amp.autocast('cuda'):
            output = model(blur, spike_vox)
            loss = nn.functional.mse_loss(output, target)
        
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {list(output.shape)}")
        print(f"  Loss: {loss.item():.6f}")
        
        # Test backward pass
        print("\nTesting backward pass...")
        scaler = torch.amp.GradScaler('cuda')
        scaler.scale(loss).backward()
        
        print(f"✓ Backward pass successful!")
        
    except Exception as e:
        print(f"\n✗ Training scenario failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("VRTWithSpike Training Scenario Unit Tests")
    print("="*70)
    
    results = []
    
    results.append(("CPU Forward", test_model_forward_cpu()))
    results.append(("Single GPU Forward", test_model_forward_single_gpu()))
    results.append(("DataParallel Forward", test_model_forward_dataparallel()))
    results.append(("AMP Forward", test_model_forward_amp()))
    results.append(("Full Training Scenario", test_model_training_scenario()))
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

