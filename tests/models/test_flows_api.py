import torch
import math

from models.optical_flow.spynet import SpyNet
from models.flows import compute_flows_2frames


def test_compute_flows_shapes():
    torch.manual_seed(0)
    b, n, c, h, w = 1, 4, 3, 64, 64
    x = torch.randn(b, n, c, h, w)

    # instantiate SpyNet with multi-level outputs (matching VRT usage)
    spynet = SpyNet(load_path=None, return_levels=[2, 3, 4, 5])

    flows_backward, flows_forward = compute_flows_2frames(spynet, x)

    assert isinstance(flows_backward, list) and isinstance(flows_forward, list)
    assert len(flows_backward) == 4 and len(flows_forward) == 4

    for i, (fb, ff) in enumerate(zip(flows_backward, flows_forward)):
        exp_h = h // (2 ** i)
        exp_w = w // (2 ** i)
        assert fb.shape == (b, n - 1, 2, exp_h, exp_w)
        assert ff.shape == (b, n - 1, 2, exp_h, exp_w)


if __name__ == "__main__":
    test_compute_flows_shapes()
    print("smoke test passed")


