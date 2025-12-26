import torch

from models.optical_flow.spynet import SpyNet
from models.flows import compute_flows_2frames


def test_spynet_forward_shapes():
    b, n, c, h, w = 1, 5, 3, 64, 64
    x = torch.randn(b, n, c, h, w)
    spynet = SpyNet(load_path=None, return_levels=[5])
    flows_b, flows_f = compute_flows_2frames(spynet, x)

    # flows_b and flows_f should be lists of tensors (one per scale)
    assert isinstance(flows_b, list) and isinstance(flows_f, list)
    for fb, ff in zip(flows_b, flows_f):
        assert fb.shape[0] == b and ff.shape[0] == b
        assert fb.shape[1] == n - 1 and ff.shape[1] == n - 1
        assert fb.shape[2] == 2 and ff.shape[2] == 2


