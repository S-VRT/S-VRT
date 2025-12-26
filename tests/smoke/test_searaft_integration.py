#!/usr/bin/env python3
"""
Simple test to verify SeaRaft integration with VRT works correctly.
"""
import torch
from models.optical_flow import create_optical_flow


def test_searaft_vrt_integration():
    """Test that SeaRaft works with VRT's expected interface."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    # Test SeaRaft with VRT-style parameters
    try:
        of = create_optical_flow(
            module='sea_raft',
            checkpoint='weights/optical_flow/Tartan-C-T-TSKH-kitti432x960-M.pth',
            device=device,
            return_levels=[2, 3, 4, 5]  # VRT expects 4 scales
        )
        print("✓ SeaRaft model created successfully")
    except Exception as e:
        print(f"✗ Failed to create SeaRaft model: {e}")
        return False

    # Test with VRT-style input (BGR [0,1])
    b, h, w = 2, 64, 64  # Small batch for testing
    frame1 = torch.rand(b, 3, h, w, device=device)  # BGR [0,1] as in VRT
    frame2 = torch.rand(b, 3, h, w, device=device)

    try:
        flows = of(frame1, frame2)
        print(f"✓ Forward pass successful, output type: {type(flows)}")

        if isinstance(flows, list):
            print(f"✓ Returned {len(flows)} flow scales")
            for i, flow in enumerate(flows):
                print(f"  Scale {i}: {flow.shape}")
                assert flow.shape[0] == b, f"Batch size mismatch: {flow.shape[0]} != {b}"
                assert flow.shape[1] == 2, f"Flow channels mismatch: {flow.shape[1]} != 2"
                assert torch.isfinite(flow).all(), f"Non-finite values in scale {i}"
        else:
            print(f"✗ Expected list of flows, got {type(flows)}")

        print("✓ SeaRaft VRT integration test passed!")
        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_searaft_vrt_integration()
    exit(0 if success else 1)
