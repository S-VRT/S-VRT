import torch
import torch.nn.functional as F
from models.optical_flow import create_optical_flow


def test_flow_consistency():
    """Test that SpyNet and SeaRaft produce consistent flow directions and shapes."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create test frames (BGR [0,1] as in VRT pipeline)
    b, c, h, w = 1, 3, 64, 64
    frame1 = torch.rand(b, c, h, w, device=device)  # BGR [0,1]
    frame2 = torch.rand(b, c, h, w, device=device)  # BGR [0,1]

    results = {}

    for backend in ['spynet', 'sea_raft']:
        print(f"\n=== Testing {backend} ===")

        try:
            of = create_optical_flow(backend, device=device)

            # Test forward pass
            flows = of(frame1, frame2)

            print(f"Output type: {type(flows)}")
            if isinstance(flows, list):
                print(f"Number of scales: {len(flows)}")
                for i, flow in enumerate(flows):
                    print(f"  Scale {i}: {flow.shape}")
                    print(f"    Finite values: {torch.isfinite(flow).all().item()}")
                    print(f"    Flow range: [{flow.min().item():.3f}, {flow.max().item():.3f}]")
            else:
                print(f"Single output: {flows.shape}")

            results[backend] = flows

        except Exception as e:
            print(f"Error with {backend}: {e}")
            results[backend] = None

    # Compare results
    if 'spynet' in results and 'sea_raft' in results and results['spynet'] is not None and results['sea_raft'] is not None:
        spy_flows = results['spynet']
        raft_flows = results['sea_raft']

        print("\n=== Comparison ===")
        print(f"SpyNet scales: {len(spy_flows) if isinstance(spy_flows, list) else 1}")
        print(f"SeaRaft scales: {len(raft_flows) if isinstance(raft_flows, list) else 1}")

        if isinstance(spy_flows, list) and isinstance(raft_flows, list) and len(spy_flows) == len(raft_flows):
            for i in range(len(spy_flows)):
                spy_flow = spy_flows[i]
                raft_flow = raft_flows[i]
                if spy_flow.shape == raft_flow.shape:
                    diff = torch.abs(spy_flow - raft_flow).mean()
                    print(f"Scale {i} - Shape: {spy_flow.shape}, Mean diff: {diff.item():.6f}")
                else:
                    print(f"Scale {i} - Shape mismatch: SpyNet {spy_flow.shape} vs SeaRaft {raft_flow.shape}")

    return results


def test_vrt_flow_integration():
    """Test optical flow integration with VRT's get_flow_2frames logic."""
    from models.flows import compute_flows_2frames

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Simulate VRT input: (b, n, c, h, w) where c=6 (RGB + Spike channels)
    b, n, c, h, w = 1, 4, 6, 64, 64
    x = torch.rand(b, n, c, h, w, device=device)

    for backend in ['spynet', 'sea_raft']:
        print(f"\n=== Testing VRT integration with {backend} ===")

        try:
            of = create_optical_flow(backend, device=device)

            # This mimics VRT's get_flow_2frames logic
            flows_backward, flows_forward = compute_flows_2frames(of, x)

            print(f"flows_backward: {len(flows_backward)} scales")
            print(f"flows_forward: {len(flows_forward)} scales")

            for i, (fb, ff) in enumerate(zip(flows_backward, flows_forward)):
                expected_h = h // (2 ** i)
                expected_w = w // (2 ** i)
                print(f"  Scale {i}: backward {fb.shape}, forward {ff.shape} (expected HxW: {expected_h}x{expected_w})")

                # Check shapes
                assert fb.shape == (b, n-1, 2, expected_h, expected_w), f"Backward flow shape mismatch at scale {i}"
                assert ff.shape == (b, n-1, 2, expected_h, expected_w), f"Forward flow shape mismatch at scale {i}"

                # Check finite values
                assert torch.isfinite(fb).all(), f"Non-finite values in backward flow at scale {i}"
                assert torch.isfinite(ff).all(), f"Non-finite values in forward flow at scale {i}"

            print(f"✓ {backend} VRT integration successful")

        except Exception as e:
            print(f"✗ {backend} VRT integration failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("Testing optical flow consistency...")
    test_flow_consistency()

    print("\n" + "="*60)
    print("Testing VRT integration...")
    test_vrt_flow_integration()

    print("\nAll tests completed!")
