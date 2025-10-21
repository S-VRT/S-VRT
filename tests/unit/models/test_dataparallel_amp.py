#!/usr/bin/env python3
"""
Simplified test for DataParallel + AMP scenario.

Tests the spike_temporal_sa module in a realistic multi-GPU training scenario.
"""
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn


def test_dataparallel_amp_scenario():
    """Test realistic DataParallel + AMP training scenario."""
    print("\n" + "="*70)
    print("DataParallel + AMP Training Scenario Test")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping...")
        return True
    
    from src.models.spike_temporal_sa import SpikeTemporalSA
    
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"\nOnly {n_gpus} GPU available, using single GPU...")
        device_ids = [0]
    else:
        print(f"\nFound {n_gpus} GPUs, using 3 GPUs...")
        device_ids = [0, 1, 2]
    
    # VRT feature pyramid channels
    channels_per_scale = [64, 128, 256, 320, 512, 640, 1280]
    
    print(f"\nCreating SpikeTemporalSA module...")
    print(f"  channels_per_scale: {channels_per_scale}")
    print(f"  heads: 8")
    
    module = SpikeTemporalSA(
        channels_per_scale=channels_per_scale,
        heads=8
    )
    
    device = torch.device(f'cuda:{device_ids[0]}')
    module = module.to(device)
    
    print(f"\nWrapping with DataParallel (GPUs: {device_ids})...")
    module = nn.DataParallel(module, device_ids=device_ids)
    module.train()
    
    # Realistic training batch size  
    batch_per_gpu = 4
    B = batch_per_gpu * len(device_ids)
    T = 5
    H, W = 128, 128
    
    print(f"\nBatch configuration:")
    print(f"  Total batch size: {B} (={batch_per_gpu} per GPU × {len(device_ids)} GPUs)")
    print(f"  Temporal length: {T}")
    print(f"  Spatial resolution: {H}×{W}")
    
    # Create multi-scale inputs (SpikeEncoder3D output format: [B, C, T, H, W])
    spike_feats = []
    for i, C in enumerate(channels_per_scale):
        scale_factor = 2 ** i
        feat = torch.randn(
            B, C, T, 
            H // scale_factor, W // scale_factor, 
            device=device
        )
        spike_feats.append(feat)
        print(f"  Scale {i} input: {list(feat.shape)}")
    
    print(f"\n{'='*70}")
    print("Test 1: Forward Pass with AMP")
    print(f"{'='*70}")
    
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = module(spike_feats)
        
        print(f"\n✓ Forward pass successful!")
        print(f"\nOutput shapes:")
        for i, out in enumerate(outputs):
            print(f"  Scale {i}: {list(out.shape)}")
        
    except Exception as e:
        print(f"\n✗ Forward pass FAILED:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'='*70}")
    print("Test 2: Backward Pass with AMP + GradScaler")
    print(f"{'='*70}")
    
    try:
        # Reset module to training mode and recreate optimizer
        module.train()
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        
        # Create inputs and targets
        spike_feats = []
        targets = []
        for i, C in enumerate(channels_per_scale):
            scale_factor = 2 ** i
            feat = torch.randn(
                B, C, T,
                H // scale_factor, W // scale_factor,
                device=device, requires_grad=True
            )
            target = torch.randn_like(feat)
            spike_feats.append(feat)
            targets.append(target)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = module(spike_feats)
            # Simple MSE loss across all scales
            loss = sum(
                nn.functional.mse_loss(out, tgt) 
                for out, tgt in zip(outputs, targets)
            )
        
        print(f"\n  Loss: {loss.item():.6f}")
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"✓ Backward pass successful!")
        
    except Exception as e:
        print(f"\n✗ Backward pass FAILED:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'='*70}")
    print("Test 3: Multiple Training Steps")
    print(f"{'='*70}")
    
    try:
        n_steps = 5
        for step in range(n_steps):
            # Create fresh inputs for each step
            spike_feats = []
            targets = []
            for i, C in enumerate(channels_per_scale):
                scale_factor = 2 ** i
                feat = torch.randn(
                    B, C, T,
                    H // scale_factor, W // scale_factor,
                    device=device, requires_grad=True
                )
                target = torch.randn_like(feat)
                spike_feats.append(feat)
                targets.append(target)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = module(spike_feats)
                loss = sum(
                    nn.functional.mse_loss(out, tgt)
                    for out, tgt in zip(outputs, targets)
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print(f"  Step {step+1}/{n_steps}: loss = {loss.item():.6f}")
        
        print(f"\n✓ Multiple training steps successful!")
        
    except Exception as e:
        print(f"\n✗ Multiple training steps FAILED:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'='*70}")
    print("✓ All tests passed!")
    print(f"{'='*70}")
    
    return True


def main():
    """Run the test."""
    success = test_dataparallel_amp_scenario()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

