import pytest
import torch
from models.optical_flow.scflow.wrapper import SCFlowWrapper

try:
    from spatial_correlation_sampler import spatial_correlation_sample as _scs  # noqa: F401
    _HAS_CORR = True
except ImportError:
    _HAS_CORR = False

requires_corr = pytest.mark.skipif(not _HAS_CORR, reason="spatial_correlation_sampler not installed")


@requires_corr
def test_scflow_wrapper_fp16_no_crash():
    """SCFlowWrapper must not crash when called inside fp16 autocast context."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wrapper = SCFlowWrapper(device='cuda')
    spk1 = torch.randint(0, 2, (1, 25, 64, 64), device='cuda').float()
    spk2 = torch.randint(0, 2, (1, 25, 64, 64), device='cuda').float()
    with torch.autocast("cuda", dtype=torch.float16):
        flows = wrapper(spk1, spk2)
    assert len(flows) == 4
    for f in flows:
        assert not torch.isnan(f).any()
        assert not torch.isinf(f).any()


@requires_corr
def test_scflow_wrapper_output_is_float32():
    """SCFlowWrapper outputs should be float32 regardless of outer autocast."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wrapper = SCFlowWrapper(device='cuda')
    spk1 = torch.randint(0, 2, (1, 25, 64, 64), device='cuda').float()
    spk2 = torch.randint(0, 2, (1, 25, 64, 64), device='cuda').float()
    with torch.autocast("cuda", dtype=torch.float16):
        flows = wrapper(spk1, spk2)
    for f in flows:
        assert f.dtype == torch.float32


@requires_corr
def test_scflow_wrapper_cpu_no_crash():
    """SCFlowWrapper must work on CPU without autocast."""
    wrapper = SCFlowWrapper(device='cpu')
    spk1 = torch.randint(0, 2, (1, 25, 32, 32)).float()
    spk2 = torch.randint(0, 2, (1, 25, 32, 32)).float()
    flows = wrapper(spk1, spk2)
    assert len(flows) == 4
    for f in flows:
        assert not torch.isnan(f).any()
