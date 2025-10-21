#!/usr/bin/env python3
"""
Unit test for SpikeTemporalSelfAttention module.

Tests the spike temporal self-attention blocks to debug shape mismatches
and ensure proper configuration with different channel dimensions.
"""
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn


def test_multihead_attention_basics():
    """Test PyTorch MultiheadAttention with different configurations."""
    print("\n" + "="*70)
    print("Test 1: PyTorch MultiheadAttention Basics")
    print("="*70)
    
    test_cases = [
        {"embed_dim": 64, "num_heads": 8, "name": "64 channels, 8 heads"},
        {"embed_dim": 128, "num_heads": 8, "name": "128 channels, 8 heads"},
        {"embed_dim": 256, "num_heads": 8, "name": "256 channels, 8 heads"},
        {"embed_dim": 320, "num_heads": 8, "name": "320 channels, 8 heads (should fail)"},
    ]
    
    for case in test_cases:
        embed_dim = case["embed_dim"]
        num_heads = case["num_heads"]
        name = case["name"]
        
        print(f"\n{name}:")
        print(f"  embed_dim={embed_dim}, num_heads={num_heads}")
        print(f"  embed_dim % num_heads = {embed_dim % num_heads}")
        
        try:
            attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )
            
            # Test with dummy input
            batch_size, seq_len = 4, 5
            x = torch.randn(batch_size, seq_len, embed_dim)
            
            with torch.no_grad():
                output, _ = attn(x, x, x, need_weights=False)
            
            print(f"  ✓ SUCCESS: output shape = {output.shape}")
            
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")


def test_temporal_self_attention_block():
    """Test TemporalSelfAttentionBlock with different channel sizes."""
    print("\n" + "="*70)
    print("Test 2: TemporalSelfAttentionBlock")
    print("="*70)
    
    from src.models.spike_temporal_sa import TemporalSelfAttentionBlock
    
    # VRT feature pyramid channels: [64, 128, 256, 320, 512, 640, 1280]
    test_cases = [
        {"dim": 64, "heads": 8, "name": "Scale 0 (64 channels)"},
        {"dim": 128, "heads": 8, "name": "Scale 1 (128 channels)"},
        {"dim": 256, "heads": 8, "name": "Scale 2 (256 channels)"},
        {"dim": 320, "heads": 8, "name": "Scale 3 (320 channels) - Expected to FAIL"},
        {"dim": 512, "heads": 8, "name": "Scale 4 (512 channels)"},
        {"dim": 640, "heads": 8, "name": "Scale 5 (640 channels) - Expected to FAIL"},
        {"dim": 1280, "heads": 8, "name": "Scale 6 (1280 channels)"},
    ]
    
    for case in test_cases:
        dim = case["dim"]
        heads = case["heads"]
        name = case["name"]
        
        print(f"\n{name}:")
        print(f"  dim={dim}, heads={heads}, divisible={dim % heads == 0}")
        
        try:
            block = TemporalSelfAttentionBlock(dim=dim, heads=heads)
            
            # Input: [B, T, C, H, W]
            B, T, C, H, W = 2, 5, dim, 16, 16
            x = torch.randn(B, T, C, H, W)
            
            print(f"  Input shape: {list(x.shape)}")
            
            with torch.no_grad():
                output = block(x)
            
            print(f"  ✓ SUCCESS: output shape = {list(output.shape)}")
            assert output.shape == x.shape, "Output shape mismatch!"
            
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")


def test_spike_temporal_sa():
    """Test full SpikeTemporalSA module."""
    print("\n" + "="*70)
    print("Test 3: SpikeTemporalSA (Full Module)")
    print("="*70)
    
    from src.models.spike_temporal_sa import SpikeTemporalSA
    
    # VRT feature pyramid channels
    channels_per_scale = [64, 128, 256, 320, 512, 640, 1280]
    
    print(f"\nAttempting to create SpikeTemporalSA with:")
    print(f"  channels_per_scale = {channels_per_scale}")
    print(f"  heads = 8")
    
    print(f"\nChannel divisibility check:")
    for i, c in enumerate(channels_per_scale):
        divisible = c % 8 == 0
        status = "✓" if divisible else "✗"
        print(f"  Scale {i}: {c} channels, divisible by 8: {status}")
    
    try:
        module = SpikeTemporalSA(
            channels_per_scale=channels_per_scale,
            heads=8
        )
        print(f"\n✓ Module created successfully!")
        
        # Test forward pass with dummy input
        # Note: SpikeEncoder3D outputs [B, C, T, H, W] format
        spike_feats = []
        B, T, H, W = 2, 5, 64, 64
        
        for i, C in enumerate(channels_per_scale):
            # Each scale has different spatial resolution
            scale_factor = 2 ** i
            # Create input in SpikeEncoder3D output format: [B, C, T, H, W]
            feat = torch.randn(B, C, T, H // scale_factor, W // scale_factor)
            spike_feats.append(feat)
            print(f"  Scale {i} input shape: {list(feat.shape)}")
        
        print(f"\nRunning forward pass...")
        with torch.no_grad():
            outputs = module(spike_feats)
        
        print(f"✓ Forward pass successful!")
        print(f"\nOutput shapes:")
        for i, out in enumerate(outputs):
            print(f"  Scale {i}: {list(out.shape)}")
        
    except Exception as e:
        print(f"\n✗ Module creation/forward FAILED:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_cuda_attention():
    """Test attention on CUDA to debug the CUDA error."""
    print("\n" + "="*70)
    print("Test 4: CUDA Attention Test")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping...")
        return
    
    device = torch.device('cuda:0')
    print(f"\nTesting on {torch.cuda.get_device_name(0)}")
    
    from src.models.spike_temporal_sa import TemporalSelfAttentionBlock
    
    # Test problematic channel sizes
    test_cases = [
        {"dim": 256, "heads": 8, "name": "256 channels (should work)"},
        {"dim": 320, "heads": 8, "name": "320 channels (problematic)"},
    ]
    
    for case in test_cases:
        dim = case["dim"]
        heads = case["heads"]
        name = case["name"]
        
        print(f"\n{name}:")
        
        if dim % heads != 0:
            print(f"  Skipping: {dim} not divisible by {heads}")
            continue
        
        try:
            block = TemporalSelfAttentionBlock(dim=dim, heads=heads).to(device)
            
            # Small test to isolate the issue
            B, T, C, H, W = 1, 5, dim, 8, 8
            x = torch.randn(B, T, C, H, W, device=device)
            
            print(f"  Input: B={B}, T={T}, C={C}, H={H}, W={W}")
            print(f"  After reshape: [BHW={B*H*W}, T={T}, C={C}]")
            
            with torch.no_grad():
                output = block(x)
            
            print(f"  ✓ SUCCESS on CUDA")
            
        except Exception as e:
            print(f"  ✗ FAILED on CUDA:")
            print(f"    {type(e).__name__}: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Spike Temporal Self-Attention Unit Tests")
    print("="*70)
    
    test_multihead_attention_basics()
    test_temporal_self_attention_block()
    test_spike_temporal_sa()
    test_cuda_attention()
    
    print("\n" + "="*70)
    print("Tests completed!")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

