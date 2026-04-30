import pytest
import torch

from scripts.analysis.fusion_attr.cam import (
    CamTargetSelection,
    build_cam_metadata,
    build_cam_scope_targets,
    compute_cam_map,
    select_cam_target,
)
from scripts.analysis.fusion_attr.stitching import (
    TileBox,
    crop_box_to_tile,
    mask_intersects_tile,
    stitch_weighted_tiles,
)


def test_crop_box_to_tile_returns_local_coordinates():
    tile = TileBox(top=10, left=20, bottom=30, right=50)
    cropped = crop_box_to_tile((25, 15, 45, 28), tile)
    assert cropped == (5, 5, 25, 18)


def test_mask_intersects_tile_detects_no_overlap():
    tile = TileBox(top=0, left=0, bottom=8, right=8)
    assert mask_intersects_tile((10, 10, 14, 14), tile) is False
    assert mask_intersects_tile((4, 4, 12, 12), tile) is True


def test_build_cam_scope_targets_separates_fullframe_and_roi():
    output = torch.ones(1, 3, 6, 6)
    gt = torch.zeros(1, 3, 6, 6)
    roi_xyxy = (2, 1, 5, 4)

    targets = build_cam_scope_targets(output=output, gt=gt, roi_xyxy=roi_xyxy)

    assert set(targets) == {"fullframe", "roi"}
    assert targets["fullframe"].item() != targets["roi"].item()


def test_select_cam_target_prefers_fused_main_for_collapsed_contract():
    record = {
        "fused_main": torch.randn(1, 4, 3, 8, 8, requires_grad=True),
        "backbone_view": torch.randn(1, 4, 3, 8, 8, requires_grad=True),
        "meta": {"frame_contract": "collapsed", "main_from_exec_rule": None, "spike_bins": 4},
    }

    selection = select_cam_target(record)

    assert isinstance(selection, CamTargetSelection)
    assert selection.tensor_name == "fused_main"
    assert selection.time_index == 2


def test_select_cam_target_uses_center_subframe_rule_for_expanded_contract():
    fused_main = torch.randn(1, 3, 3, 8, 8, requires_grad=True)
    backbone_view = torch.randn(1, 12, 3, 8, 8, requires_grad=True)
    record = {
        "fused_main": fused_main,
        "backbone_view": backbone_view,
        "meta": {"frame_contract": "expanded", "main_from_exec_rule": "center_subframe", "spike_bins": 4},
    }

    selection = select_cam_target(record)

    assert selection.tensor_name == "backbone_view"
    assert selection.time_index == 1 * 4 + 2


def test_compute_cam_map_dispatches_hirescam():
    activation = torch.ones(1, 3, 4, 4, requires_grad=True)
    target = activation[:, :, :2, :2].sum()

    out = compute_cam_map(activation=activation, target=target, method="hirescam", time_index=None)

    assert out.shape == (4, 4)
    assert out.max().item() > 0.0


def test_stitch_weighted_tiles_smooths_overlap_boundaries():
    tile_a = torch.ones(1, 1, 8, 8)
    tile_b = torch.ones(1, 1, 8, 8) * 3.0
    stitched = stitch_weighted_tiles(
        canvas_shape=(1, 1, 8, 12),
        tiles=[
            (tile_a, TileBox(top=0, left=0, bottom=8, right=8)),
            (tile_b, TileBox(top=0, left=4, bottom=8, right=12)),
        ],
    )

    seam_profile = stitched[0, 0, 4]
    assert seam_profile[3].item() < seam_profile[6].item()
    assert seam_profile.min().item() >= 0.0
    assert seam_profile.max().item() <= 3.0


def test_build_cam_metadata_records_repaired_contract():
    selection = CamTargetSelection(
        activation=torch.randn(1, 3, 4, 4),
        tensor_name="backbone_view",
        time_index=6,
        frame_contract="expanded",
        main_from_exec_rule="center_subframe",
        spike_bins=4,
    )

    metadata = build_cam_metadata(
        requested_method="hirescam",
        effective_method="hirescam",
        default_scope="fullframe",
        scopes_exported=["fullframe", "roi"],
        selection=selection,
        analysis_crop_size=256,
        analysis_tile_stride=128,
        stitch_weight="hann",
        roi_xyxy=(4, 5, 20, 24),
    )

    assert metadata["cam_default_scope"] == "fullframe"
    assert metadata["cam_scopes_exported"] == ["fullframe", "roi"]
    assert metadata["cam_target_tensor"] == "backbone_view"
    assert metadata["analysis_tile_overlap"] == 128
