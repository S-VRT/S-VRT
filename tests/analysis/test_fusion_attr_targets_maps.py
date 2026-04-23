import pytest
import torch

from scripts.analysis.fusion_attr.io import AnalysisSample
from scripts.analysis.fusion_attr.maps import (
    compute_error_map,
    compute_fusion_delta,
    normalize_map,
    reduce_to_2d,
)
from scripts.analysis.fusion_attr.perturb import perturb_spike
from scripts.analysis.fusion_attr.targets import build_box_mask, masked_charbonnier_target


def _sample():
    return AnalysisSample(
        clip="clip",
        frame="000001",
        frame_index=0,
        mask_type="box",
        xyxy=(1, 1, 3, 4),
        mask_label="box",
        reason="unit test",
    )


def test_build_box_mask_marks_expected_region():
    mask = build_box_mask(_sample(), height=5, width=6, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 5, 6)
    assert mask.sum().item() == 6
    assert mask[0, 0, 1:4, 1:3].sum().item() == 6


def test_masked_charbonnier_target_is_negative_loss():
    output = torch.ones(1, 3, 5, 6)
    gt = torch.zeros(1, 3, 5, 6)
    mask = build_box_mask(_sample(), 5, 6, output.device)
    target = masked_charbonnier_target(output, gt, mask, eps=1e-6)
    assert target.item() < 0


def test_reduce_to_2d_handles_5d_center_frame():
    tensor = torch.zeros(1, 5, 3, 4, 4)
    tensor[:, 2] = 2.0
    reduced = reduce_to_2d(tensor)
    assert reduced.shape == (4, 4)
    assert reduced.max().item() == pytest.approx(2.0 * (3 ** 0.5), rel=1e-5)


def test_normalize_map_percentile_clips_to_unit_range():
    values = torch.tensor([[0.0, 1.0], [2.0, 100.0]])
    out = normalize_map(values, low=0, high=100)
    assert out.min().item() == 0.0
    assert out.max().item() == 1.0


def test_compute_fusion_delta_uses_matching_shape():
    fused = torch.ones(1, 3, 4, 4)
    reference = torch.zeros(1, 3, 4, 4)
    delta = compute_fusion_delta(fused, reference)
    assert delta.shape == (4, 4)
    assert delta.max().item() == pytest.approx(3 ** 0.5, rel=1e-5)


def test_compute_error_map_returns_mean_channel_abs_error():
    output = torch.ones(1, 3, 4, 4)
    gt = torch.zeros(1, 3, 4, 4)
    error = compute_error_map(output, gt)
    assert error.shape == (4, 4)
    assert error.max().item() == 1.0


def test_perturb_spike_zero_and_temporal_drop():
    spike = torch.ones(1, 4, 2, 3, 3)
    assert perturb_spike(spike, "zero").sum().item() == 0.0
    dropped = perturb_spike(spike, "temporal-drop")
    assert dropped[:, 2].sum().item() == 0.0
    assert dropped[:, 0].sum().item() > 0.0
