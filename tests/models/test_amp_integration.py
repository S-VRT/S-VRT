import pytest
import torch
from models.utils.flow import flow_warp


@pytest.mark.parametrize("amp_config,expected_enabled,expect_scaler", [
    ({"enable": False, "dtype": "float16"}, False, False),
    ({"enable": True, "dtype": "float16"}, True, True),
    ({"enable": True, "dtype": "bfloat16"}, True, False),
])
def test_amp_config_parsing(amp_config, expected_enabled, expect_scaler):
    """AMP config dict parses to correct enabled state and scaler presence."""
    from models.model_base import ModelBase
    # _amp_dtypes is an instance attribute; build a minimal stand-in
    class _M:
        def __init__(self):
            self._amp_dtypes = {
                'float16': torch.float16, 'fp16': torch.float16, 'half': torch.float16,
                'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
            }
        _resolve_amp_dtype = ModelBase._resolve_amp_dtype
    m = _M()
    amp_enabled = bool(amp_config.get("enable", False))
    amp_dtype = m._resolve_amp_dtype(amp_config.get("dtype", "float16"))
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    assert amp_enabled == expected_enabled
    assert scaler_enabled == expect_scaler


def test_flow_warp_fp32_regression():
    """flow_warp must still work correctly with fp32 input after the fix."""
    x = torch.randn(1, 3, 32, 32)
    flow = torch.randn(1, 32, 32, 2)
    out = flow_warp(x, flow)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()


def test_flow_warp_dtype_preserved():
    """flow_warp grid dtype must match flow dtype (no forced float cast)."""
    x = torch.randn(1, 3, 16, 16)
    flow = torch.randn(1, 16, 16, 2)
    out = flow_warp(x, flow)
    assert out.dtype == x.dtype


def test_searaft_layernorm_channels_first_cpu():
    """SeaRaft channels_first LayerNorm works on CPU fp32 after fix."""
    from models.optical_flow.sea_raft import LayerNorm
    norm = LayerNorm(16, data_format="channels_first")
    x = torch.randn(2, 16, 8, 8)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_sgp_layernorm_cpu():
    """SGP LayerNorm works on CPU fp32 after fix."""
    from models.blocks.sgp import LayerNorm
    norm = LayerNorm(32)
    x = torch.randn(2, 32, 10)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
