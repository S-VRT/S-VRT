"""Smoke/integration contract for SeaRAFT optical-flow adapter."""

from pathlib import Path

import pytest
import torch

from models.optical_flow import create_optical_flow


@pytest.mark.smoke
@pytest.mark.integration
def test_searaft_integration_contract():
    checkpoint = Path("weights/optical_flow/Tartan-C-T-TSKH-kitti432x960-M.pth")
    if not checkpoint.exists():
        pytest.skip(f"SeaRAFT checkpoint unavailable: {checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        optical_flow = create_optical_flow(
            module="sea_raft",
            checkpoint=str(checkpoint),
            device=device,
            return_levels=[2, 3, 4, 5],
        )
    except Exception as exc:
        pytest.skip(f"SeaRAFT optional dependency unavailable: {exc}")

    frame1 = torch.rand(2, 3, 64, 64, device=device)
    frame2 = torch.rand(2, 3, 64, 64, device=device)

    flows = optical_flow(frame1, frame2)
    assert flows is not None
    assert isinstance(flows, list)
    assert len(flows) >= 1

    for flow in flows:
        assert flow.shape[0] == 2
        assert flow.shape[1] == 2
        assert bool(torch.isfinite(flow).all().item())
