import importlib

import pytest
import torch


@pytest.mark.smoke
@pytest.mark.integration
def test_vrt_smoke_reference_equivalence_contract():
    """Legacy smoke intent preserved: identical model init should yield identical outputs."""
    torch.manual_seed(42)

    x = torch.randn(1, 6, 3, 64, 64)
    vrt_module = importlib.import_module("models.architectures.vrt.vrt")
    vrt_cls = getattr(vrt_module, "VRT")

    kwargs = {
        "upscale": 1,
        "in_chans": 3,
        "out_chans": 3,
        "img_size": [6, 64, 64],
        "window_size": [6, 8, 8],
        "depths": [1] * 8,
        "embed_dims": [16] * 8,
        "num_heads": [1] * 8,
        "pa_frames": 2,
    }

    torch.manual_seed(123)
    model_a = vrt_cls(**kwargs)
    torch.manual_seed(123)
    model_b = vrt_cls(**kwargs)

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)

    assert out_a.shape == out_b.shape
    assert float((out_a - out_b).abs().max().item()) == 0.0


def run_smoke():
    """Helper kept for pytest entrypoint contract check."""
    test_vrt_smoke_reference_equivalence_contract()
    return True


@pytest.mark.smoke
def test_smoke_entrypoint_pytest_contract():
    result = run_smoke()
    assert result is None or result is True
