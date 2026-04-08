import pytest
import torch

from models.fusion.adapters.middle import MiddleFusionAdapter
from models.fusion.factory import create_fusion_operator


def test_middle_replace_mode():
    op = create_fusion_operator("concat", 24, 24, 24, {})
    adapter = MiddleFusionAdapter(operator=op, mode="replace", inject_stages=[1, 3])
    x = torch.randn(1, 24, 6, 8, 8)
    spike_ctx = torch.randn(1, 24, 6, 8, 8)
    y = adapter(stage_idx=1, x=x, spike_ctx=spike_ctx)
    assert y.shape == x.shape


def test_middle_skip_non_injected_stage():
    op = create_fusion_operator("concat", 24, 24, 24, {})
    adapter = MiddleFusionAdapter(operator=op, mode="replace", inject_stages=[2])
    x = torch.randn(1, 24, 6, 8, 8)
    spike_ctx = torch.randn(1, 24, 6, 8, 8)
    y = adapter(stage_idx=1, x=x, spike_ctx=spike_ctx)
    assert torch.equal(x, y)


def test_middle_rejects_spike_channel_mismatch():
    op = create_fusion_operator("concat", 24, 12, 24, {})
    adapter = MiddleFusionAdapter(operator=op, mode="replace", inject_stages=[1])
    x = torch.randn(1, 24, 6, 8, 8)
    spike_ctx = torch.randn(1, 8, 6, 8, 8)
    with pytest.raises(ValueError, match="spike_ctx channels"):
        adapter(stage_idx=1, x=x, spike_ctx=spike_ctx)
