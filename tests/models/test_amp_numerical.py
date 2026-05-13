import pytest
import torch
from models.optical_flow.sea_raft import LayerNorm as SeaRaftLayerNorm
from models.blocks.sgp import LayerNorm as SGPLayerNorm


def test_searaft_layernorm_channels_first_fp16():
    """channels_first LayerNorm must not produce NaN/inf with fp16 input."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    norm = SeaRaftLayerNorm(32, data_format="channels_first").cuda()
    x = torch.randn(2, 32, 64, 64, device='cuda', dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        out = norm(x)
    assert out.dtype == torch.float16
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_searaft_layernorm_channels_last_fp16():
    """channels_last LayerNorm (via F.layer_norm) must work with fp16."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    norm = SeaRaftLayerNorm(32, data_format="channels_last").cuda()
    x = torch.randn(2, 64, 64, 32, device='cuda', dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        out = norm(x)
    assert not torch.isnan(out).any()


def test_sgp_layernorm_fp16():
    """SGP custom LayerNorm must not produce NaN/inf with fp16 input."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    norm = SGPLayerNorm(64).cuda()
    x = torch.randn(2, 64, 16, device='cuda', dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        out = norm(x)
    assert out.dtype == torch.float16
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_searaft_layernorm_channels_first_cpu_fp32():
    """channels_first LayerNorm must work correctly on CPU with fp32."""
    norm = SeaRaftLayerNorm(16, data_format="channels_first")
    x = torch.randn(2, 16, 8, 8)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_sgp_layernorm_cpu_fp32():
    """SGP LayerNorm must work correctly on CPU with fp32."""
    norm = SGPLayerNorm(32)
    x = torch.randn(2, 32, 10)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
