"""Compatibility smoke tests for flow_composition module (alias of flow_utils)."""
from mmvrt.models.motion.flow_composition import get_flow_4frames, get_flow_6frames
import torch


def _make_dummy_flows(b=1, n=5, h=8, w=8, scales=1):
    flows = []
    for _ in range(scales):
        flows.append(torch.rand((b, n - 1, 2, h, w), dtype=torch.float32))
    return flows


def test_flow_comp_shapes():
    b, n, h, w = 1, 5, 8, 8
    ff = _make_dummy_flows(b=b, n=n, h=h, w=w, scales=1)
    fb = _make_dummy_flows(b=b, n=n, h=h, w=w, scales=1)
    fb2, ff2 = get_flow_4frames(ff, fb)
    fb3, ff3 = get_flow_6frames(ff, fb, ff2, fb2)
    assert fb2 and ff2 and fb3 and ff3
    print("flow_composition smoke passed")


if __name__ == "__main__":
    test_flow_comp_shapes()


