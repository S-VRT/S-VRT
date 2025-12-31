#!/usr/bin/env python3
"""
Unit tests for DCN modules (DCNv2 and DCNv4) to verify forward/backward compatibility.
"""

import torch
import torch.nn as nn
import pytest
from torch.autograd import gradcheck


def test_dcnv2_forward_backward():
    """Test DCNv2PackFlowGuided forward and backward pass."""
    from models.blocks.dcn import DCNv2PackFlowGuided

    # Test parameters
    batch_size, channels, height, width = 2, 64, 32, 32
    pa_frames = 2

    # Create module
    dcn = DCNv2PackFlowGuided(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        padding=1,
        deformable_groups=8,
        pa_frames=pa_frames
    )

    # Create test inputs
    x = torch.randn(batch_size, channels, height, width, requires_grad=True)
    x_flow_warpeds = [torch.randn_like(x) for _ in range(pa_frames // 2)]
    x_current = torch.randn_like(x)
    flows = [torch.randn(batch_size, 2, height, width) for _ in range(pa_frames // 2)]

    # Forward pass
    output = dcn(x, x_flow_warpeds, x_current, flows)

    # Check output shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients
    assert x.grad is not None, "Input gradient should not be None"
    assert x.grad.shape == x.shape, f"Input gradient shape mismatch: {x.grad.shape} vs {x.shape}"

    print("DCNv2 forward/backward test passed!")


def test_dcnv4_forward_backward():
    """Test DCNv4PackFlowGuided forward and backward pass."""
    from models.blocks.dcn import DCNv4PackFlowGuided

    # Skip test if DCNv4 is not available
    try:
        dcn_module = DCNv4PackFlowGuided
    except ImportError as e:
        pytest.skip(f"DCNv4 not available: {e}")
        return

    # Test parameters - adjusted for DCNv4 requirements (_d_per_group % 16 == 0)
    batch_size, channels, height, width = 2, 128, 32, 32  # channels=128, group=4 -> _d_per_group=32
    pa_frames = 2

    # Create test inputs on GPU
    device = torch.device('cuda')
    x = torch.randn(batch_size, channels, height, width, requires_grad=True, device=device)
    x_flow_warpeds = [torch.randn_like(x) for _ in range(pa_frames // 2)]
    x_current = torch.randn_like(x)
    flows = [torch.randn(batch_size, 2, height, width, device=device) for _ in range(pa_frames // 2)]

    # Create module and move to GPU
    dcn = dcn_module(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        padding=1,
        deformable_groups=4,  # 128//4=32, 32%16==0
        pa_frames=pa_frames
    ).to(device)

    # Forward pass
    output = dcn(x, x_flow_warpeds, x_current, flows)

    # Check output shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients
    assert x.grad is not None, "Input gradient should not be None"
    assert x.grad.shape == x.shape, f"Input gradient shape mismatch: {x.grad.shape} vs {x.shape}"

    print("DCNv4 forward/backward test passed!")


def test_dcn_factory():
    """Test the DCN factory function."""
    from models.blocks.dcn import get_deformable_module

    # Test DCNv2 selection
    opt_dcnv2 = {'netG': {'dcn_type': 'DCNv2'}}
    DCNClass = get_deformable_module(opt_dcnv2)
    assert DCNClass.__name__ == 'DCNv2PackFlowGuided', f"Expected DCNv2PackFlowGuided, got {DCNClass.__name__}"

    # Test DCNv4 selection (may skip if not available)
    opt_dcnv4 = {'netG': {'dcn_type': 'DCNv4'}}
    DCNClass = get_deformable_module(opt_dcnv4)
    if DCNClass.__name__ == 'DCNv4PackFlowGuided':
        print("DCNv4 available and correctly selected")
    else:
        # DCNv4 not available, should default to DCNv2
        assert DCNClass.__name__ == 'DCNv2PackFlowGuided', f"Expected fallback to DCNv2PackFlowGuided, got {DCNClass.__name__}"

    # Test default (no dcn_type specified)
    opt_default = {}
    DCNClass = get_deformable_module(opt_default)
    assert DCNClass.__name__ == 'DCNv2PackFlowGuided', f"Expected default DCNv2PackFlowGuided, got {DCNClass.__name__}"

    print("DCN factory test passed!")


def test_dcn_numerical_stability():
    """Test numerical stability of DCN operations."""
    from models.blocks.dcn import DCNv2PackFlowGuided

    # Test with small inputs to check for numerical issues
    batch_size, channels, height, width = 1, 32, 16, 16
    pa_frames = 2

    dcn = DCNv2PackFlowGuided(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        padding=1,
        deformable_groups=4,
        pa_frames=pa_frames
    )

    # Test with small values
    x = torch.randn(batch_size, channels, height, width) * 0.01
    x_flow_warpeds = [torch.randn_like(x) * 0.01 for _ in range(pa_frames // 2)]
    x_current = torch.randn_like(x) * 0.01
    flows = [torch.randn(batch_size, 2, height, width) * 0.01 for _ in range(pa_frames // 2)]

    # Forward pass
    output = dcn(x, x_flow_warpeds, x_current, flows)

    # Check for NaN or Inf
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"

    # Check output range is reasonable
    assert output.abs().mean() < 10.0, f"Output values too large: {output.abs().mean()}"

    print("DCN numerical stability test passed!")


if __name__ == '__main__':
    # Run tests
    test_dcn_factory()
    test_dcnv2_forward_backward()
    test_dcn_numerical_stability()

    try:
        test_dcnv4_forward_backward()
    except Exception as e:
        print(f"DCNv4 test skipped or failed: {e}")

    print("All tests completed!")

