#!/usr/bin/env python3
"""
Test script to check optical flow output order for SeaRAFT vs SpyNet
"""

import torch
import sys
import os
sys.path.append('.')

def test_optical_flow_order():
    from models.optical_flow import create_optical_flow

    print("Testing optical flow output order...")

    # Create test input (batch_size=1, frames=6, channels=3, height=160, width=160)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 6, 3, 160, 160).to(device)

    # Extract RGB channels for optical flow (VRT does this)
    x_flow = x[:, :, :3, :, :]  # Take first 3 channels (RGB)
    b, n, c, h, w = x_flow.size()

    # Reshape for optical flow input: (b*(n-1), c, h, w)
    x_1 = x_flow[:, :-1, :, :, :].reshape(-1, c, h, w)  # (5, 3, 160, 160)
    x_2 = x_flow[:, 1:, :, :, :].reshape(-1, c, h, w)   # (5, 3, 160, 160)

    print(f"Input shapes: x_flow={x_flow.shape}, x_1={x_1.shape}, x_2={x_2.shape}")

    # Test SeaRAFT
    print("\n=== Testing SeaRAFT ===")
    try:
        sea_raft = create_optical_flow('sea_raft', device=device)
        flows_backward_sea = sea_raft(x_1, x_2)
        flows_forward_sea = sea_raft(x_2, x_1)

        print(f"SeaRAFT flows_backward type: {type(flows_backward_sea)}")
        if isinstance(flows_backward_sea, list):
            print(f"SeaRAFT flows_backward lengths: {len(flows_backward_sea)}")
            for i, flow in enumerate(flows_backward_sea):
                print(f"  flow[{i}] shape: {flow.shape}")
        else:
            print(f"SeaRAFT flows_backward shape: {flows_backward_sea.shape}")

    except Exception as e:
        print(f"SeaRAFT test failed: {e}")

    # Test SpyNet (with correct return_levels like VRT uses)
    print("\n=== Testing SpyNet ===")
    try:
        spynet = create_optical_flow('spynet', device=device, return_levels=[2, 3, 4, 5])
        flows_backward_spy = spynet(x_1, x_2)
        flows_forward_spy = spynet(x_2, x_1)

        print(f"SpyNet flows_backward type: {type(flows_backward_spy)}")
        if isinstance(flows_backward_spy, list):
            print(f"SpyNet flows_backward lengths: {len(flows_backward_spy)}")
            for i, flow in enumerate(flows_backward_spy):
                print(f"  flow[{i}] shape: {flow.shape}")
        else:
            print(f"SpyNet flows_backward shape: {flows_backward_spy.shape}")

    except Exception as e:
        print(f"SpyNet test failed: {e}")

    # Test VRT processing logic
    print("\n=== Testing VRT Processing Logic ===")

    def simulate_vrt_processing(flows_backward, flows_forward, module_name):
        print(f"\n--- VRT processing for {module_name} ---")

        # Check if SeaRAFT (has 'sea' in class name)
        is_sea_raft = 'sea' in module_name.lower()

        if is_sea_raft:
            print("Using SeaRAFT branch")
            # SeaRAFT returns flows in correct format
            flows_backward_processed = flows_backward
            flows_forward_processed = flows_forward
        else:
            print("Using SpyNet branch (with reverse)")
            # SpyNet needs reverse
            flows_backward_processed = list(reversed(flows_backward))
            flows_forward_processed = list(reversed(flows_forward))

        print("Processed flows:")
        for i, flow in enumerate(flows_backward_processed):
            expected_shape = (b, n-1, 2, h // (2 ** i), w // (2 ** i))
            actual_shape = flow.shape
            match = "✓" if actual_shape == expected_shape else "✗"
            print(f"  level{i}: expected {expected_shape}, actual {actual_shape} {match}")

        return flows_backward_processed, flows_forward_processed

    # Test both
    if 'flows_backward_sea' in locals():
        simulate_vrt_processing(flows_backward_sea, flows_forward_sea, "SeaRAFT")

    if 'flows_backward_spy' in locals():
        simulate_vrt_processing(flows_backward_spy, flows_forward_spy, "SpyNet")

if __name__ == "__main__":
    test_optical_flow_order()
