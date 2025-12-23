"""Simple smoke tests for flow composition utilities."""
import torch

from mmvrt.models.motion.flow_composition import get_flow_4frames, get_flow_6frames


def _make_dummy_flows(b=1, n=5, h=8, w=8, scales=1):
    """Create lists of dummy flow tensors (per-scale)."""
    d = n - 1
    flows = []
    for _ in range(scales):
        # shape: (B, D, 2, H, W)
        flows.append(torch.rand((b, d, 2, h, w), dtype=torch.float32))
    return flows


def test_get_flow_4frames():
    b, n, h, w = 1, 5, 8, 8
    flows_forward = _make_dummy_flows(b=b, n=n, h=h, w=w, scales=1)
    flows_backward = _make_dummy_flows(b=b, n=n, h=h, w=w, scales=1)

    fb2, ff2 = get_flow_4frames(flows_forward, flows_backward)

    # Expect each returned tensor to have D-1 frames in temporal dim
    assert len(fb2) == len(flows_backward)
    assert len(ff2) == len(flows_forward)
    for t in fb2 + ff2:
        assert t.dim() == 5
        assert t.shape[1] == (n - 1) - 1


def test_get_flow_6frames():
    b, n, h, w = 1, 5, 8, 8
    # original (t,t+1)
    flows_forward = _make_dummy_flows(b=b, n=n, h=h, w=w, scales=1)
    flows_backward = _make_dummy_flows(b=b, n=n, h=h, w=w, scales=1)

    # build (t,t+2) using the helper above (reuse implementation to create valid shapes)
    fb2, ff2 = get_flow_4frames(flows_forward, flows_backward)

    fb3, ff3 = get_flow_6frames(flows_forward, flows_backward, ff2, fb2)

    assert len(fb3) == len(flows_backward)
    assert len(ff3) == len(flows_forward)
    for t in fb3 + ff3:
        assert t.dim() == 5


if __name__ == "__main__":
    test_get_flow_4frames()
    test_get_flow_6frames()
    print("flow_utils smoke tests passed")


