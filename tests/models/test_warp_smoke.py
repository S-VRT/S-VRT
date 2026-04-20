import pytest
import torch

from models.optical_flow.spynet import SpyNet
from models.flows import compute_flows_2frames
from models.utils.flow import flow_warp


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


def test_flow_warp_fp16():
    """flow_warp must not crash with fp16 input under autocast."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(1, 3, 64, 64, device='cuda', dtype=torch.float16)
    flow = torch.randn(1, 64, 64, 2, device='cuda', dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        out = flow_warp(x, flow)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_flow_warp_fp32_unchanged():
    """flow_warp must still work correctly with fp32 input after the fix."""
    x = torch.randn(1, 3, 32, 32)
    flow = torch.randn(1, 32, 32, 2)
    out = flow_warp(x, flow)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
