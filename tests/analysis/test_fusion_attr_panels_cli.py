import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.analysis.fusion_attr.io import save_gray_map_png, save_rgb_tensor_png
from scripts.analysis.fusion_attr.panels import make_six_column_panel


def test_save_rgb_tensor_png_writes_bgr_file(tmp_path: Path):
    tensor = torch.zeros(3, 4, 5)
    tensor[0] = 1.0
    path = tmp_path / "rgb.png"
    save_rgb_tensor_png(path, tensor)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img.shape == (4, 5, 3)
    assert int(img[:, :, 2].max()) == 255


def test_save_gray_map_png_writes_uint8_file(tmp_path: Path):
    path = tmp_path / "map.png"
    save_gray_map_png(path, torch.ones(4, 5))
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (4, 5)


def test_make_six_column_panel_writes_panel(tmp_path: Path):
    images = {
        "Blurry RGB": np.zeros((12, 16, 3), dtype=np.uint8),
        "Spike cue": np.ones((12, 16, 3), dtype=np.uint8) * 20,
        "Restored": np.ones((12, 16, 3), dtype=np.uint8) * 40,
        "Error reduction": np.ones((12, 16, 3), dtype=np.uint8) * 60,
        "Attribution heatmap": np.ones((12, 16, 3), dtype=np.uint8) * 80,
        "Fusion-specific map": np.ones((12, 16, 3), dtype=np.uint8) * 100,
    }
    out = tmp_path / "panel.png"
    make_six_column_panel(out, images)
    panel = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert panel is not None
    assert panel.shape[0] > 12
    assert panel.shape[1] > 16 * 5


def test_fusion_attribution_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/analysis/fusion_attribution.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--opt" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--samples" in result.stdout


def test_fusion_attribution_cli_dry_run_writes_manifest(tmp_path: Path):
    opt = tmp_path / "opt.json"
    samples = tmp_path / "samples.json"
    out = tmp_path / "out"
    opt.write_text('{"model":"vrt","netG":{"fusion":{"operator":"gated","placement":"early","mode":"replace"}}}', encoding="utf-8")
    samples.write_text(
        '{"samples":[{"clip":"clip","frame":"000001","frame_index":0,"mask":{"type":"box","xyxy":[0,0,2,2]},"reason":"unit"}]}',
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/fusion_attribution.py",
            "--opt",
            str(opt),
            "--checkpoint",
            "missing.pth",
            "--samples",
            str(samples),
            "--out",
            str(out),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry run complete" in result.stdout.lower()
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint"] == "missing.pth"
    assert manifest["num_samples"] == 1


from scripts.analysis.fusion_attribution import (
    build_parser,
    resolve_cam_default_scope,
    resolve_tile_stride,
    select_center_frame_tensor,
)


def test_select_center_frame_tensor_handles_5d_and_4d():
    video = torch.zeros(1, 5, 3, 4, 4)
    video[:, 2] = 7
    image = torch.ones(1, 3, 4, 4)
    assert select_center_frame_tensor(video).max().item() == 7
    assert select_center_frame_tensor(image).max().item() == 1


def test_resolve_tile_stride_only_defaults_when_omitted():
    assert resolve_tile_stride(256, None) == 128
    assert resolve_tile_stride(256, 0) == 0


def test_resolve_cam_default_scope_uses_exported_scope():
    assert resolve_cam_default_scope(["fullframe", "roi"]) == "fullframe"
    assert resolve_cam_default_scope(["roi"]) == "roi"


import pytest

from scripts.analysis.fusion_attr.pca import (
    pca_feature_heatmap,
    pca_variance_ratio,
)


def test_pca_variance_ratio_sums_to_one_for_non_degenerate_input():
    feat = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    ratio = pca_variance_ratio(feat)
    assert ratio.ndim == 1
    assert ratio.sum().item() == pytest.approx(1.0, rel=1e-5)


def test_pca_feature_heatmap_returns_spatial_map():
    feat = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    heatmap = pca_feature_heatmap(feat)
    assert heatmap.shape == (3, 4)


def test_fusion_attribution_cli_help_mentions_ig_and_pca():
    result = subprocess.run(
        [sys.executable, "scripts/analysis/fusion_attribution.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--ig-steps" in result.stdout
    assert "--save-ig" in result.stdout
    assert "--save-pca" in result.stdout


def test_fusion_attribution_cli_help_mentions_cam_scopes():
    help_text = build_parser().format_help()
    assert "--cam-scopes" in help_text
    assert "--stitch-weight" in help_text


def test_fusion_attribution_cli_help_mentions_explicit_roi_cam_outputs():
    help_text = build_parser().format_help()
    assert "cam_roi_target_fullframe" in help_text
    assert "cam_roi_crop" in help_text


def test_fusion_attribution_cli_dry_run_manifest_keeps_requested_cam_method(tmp_path: Path):
    opt = tmp_path / "opt.json"
    samples = tmp_path / "samples.json"
    out = tmp_path / "out"
    opt.write_text('{"model":"vrt","netG":{"fusion":{"operator":"gated","placement":"early","mode":"replace"}}}', encoding="utf-8")
    samples.write_text(
        '{"samples":[{"clip":"clip","frame":"000001","frame_index":0,"mask":{"type":"box","xyxy":[0,0,2,2]},"reason":"unit"}]}',
        encoding="utf-8",
    )
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/analysis/fusion_attribution.py",
            "--opt",
            str(opt),
            "--checkpoint",
            "missing.pth",
            "--samples",
            str(samples),
            "--out",
            str(out),
            "--cam-method",
            "hirescam",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["target"] == "masked_charbonnier"
    assert manifest["cam_method"] == "hirescam"
    assert manifest["cam_method"] in {"gradcam", "hirescam", "fallback"}
    assert manifest["num_samples"] == 1
