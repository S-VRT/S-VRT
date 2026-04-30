# Fusion Attribution CAM Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the offline fusion attribution pipeline so CAM exports have correct full-frame and ROI semantics, correct fusion/time alignment, real CAM method dispatch, and overlap-weighted tile stitching without obvious block seams.

**Architecture:** Keep `scripts/analysis/fusion_attribution.py` as the CLI entry point, but move the new CAM-selection and tile-stitching logic into focused helper modules under `scripts/analysis/fusion_attr/`. Add one focused repair test file for semantics and stitching behavior, then update the CLI/export tests to assert the new filenames and metadata contract.

**Tech Stack:** Python, PyTorch, NumPy, OpenCV, argparse, JSON, pytest

---

## File Structure

- Create: `scripts/analysis/fusion_attr/cam.py`
  Own CAM scope definitions, CAM method dispatch, fusion target tensor selection, and CAM metadata helpers.
- Create: `scripts/analysis/fusion_attr/stitching.py`
  Own overlap windows, weighted accumulation, ROI box cropping, and tile-overlap helpers.
- Modify: `scripts/analysis/fusion_attr/probes.py`
  Preserve structured adapter outputs so attribution can choose `fused_main` vs `backbone_view` explicitly.
- Modify: `scripts/analysis/fusion_attr/maps.py`
  Add HiResCAM-style reduction and shared activation-to-2D helpers used by `cam.py`.
- Modify: `scripts/analysis/fusion_attribution.py`
  Wire the helpers into the CLI flow, export both CAM scopes, rename outputs, and write the new metadata.
- Create: `tests/analysis/test_fusion_attr_cam_repair.py`
  Focused unit and numerical tests for scope semantics, alignment, CAM dispatch, and weighted stitching.
- Modify: `tests/analysis/test_fusion_attr_panels_cli.py`
  Update CLI/export expectations to the repaired filenames and metadata fields.

### Task 1: Add Failing Tests For CAM Scope Semantics And Weighted Stitching

**Files:**
- Create: `tests/analysis/test_fusion_attr_cam_repair.py`

- [ ] **Step 1: Write the failing tests**

Add this new file:

```python
import pytest
import torch

from scripts.analysis.fusion_attr.cam import (
    CamTargetSelection,
    build_cam_scope_targets,
    build_cam_metadata,
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
    assert selection.time_index == 2 * 4 + 2


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/analysis/test_fusion_attr_cam_repair.py -q
```

Expected: FAIL with import errors for `scripts.analysis.fusion_attr.cam` and `scripts.analysis.fusion_attr.stitching`.

- [ ] **Step 3: Commit the failing-test checkpoint**

```bash
git add tests/analysis/test_fusion_attr_cam_repair.py
git commit -m "test(analysis): add failing cam repair tests"
```

### Task 2: Implement CAM Selection, Scope Targets, And Method Dispatch

**Files:**
- Create: `scripts/analysis/fusion_attr/cam.py`
- Modify: `scripts/analysis/fusion_attr/maps.py`
- Modify: `scripts/analysis/fusion_attr/probes.py`
- Test: `tests/analysis/test_fusion_attr_cam_repair.py`

- [ ] **Step 1: Implement the new CAM helper module**

Create `scripts/analysis/fusion_attr/cam.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch

from .maps import gradcam_from_activation, hirescam_from_activation
from .targets import masked_charbonnier_target


@dataclass(frozen=True)
class CamTargetSelection:
    activation: torch.Tensor
    tensor_name: str
    time_index: int | None
    frame_contract: str
    main_from_exec_rule: str | None
    spike_bins: int


def fullframe_charbonnier_target(output: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = output - gt
    return -torch.sqrt(diff * diff + eps).mean()


def build_cam_scope_targets(
    output: torch.Tensor,
    gt: torch.Tensor,
    roi_xyxy: tuple[int, int, int, int] | None,
) -> dict[str, torch.Tensor]:
    fullframe = fullframe_charbonnier_target(output, gt)
    targets = {"fullframe": fullframe}
    if roi_xyxy is not None:
        x1, y1, x2, y2 = roi_xyxy
        mask = torch.zeros(1, 1, output.shape[-2], output.shape[-1], device=output.device)
        mask[:, :, y1:y2, x1:x2] = 1.0
        targets["roi"] = masked_charbonnier_target(output, gt, mask)
    return targets


def select_cam_target(record: dict) -> CamTargetSelection:
    fused_main = record["fused_main"]
    backbone_view = record["backbone_view"]
    meta = dict(record.get("meta") or {})
    frame_contract = str(meta.get("frame_contract", "collapsed"))
    main_from_exec_rule = meta.get("main_from_exec_rule")
    spike_bins = int(meta.get("spike_bins", 1) or 1)

    if frame_contract == "expanded" and main_from_exec_rule == "center_subframe":
        frame_idx = fused_main.shape[1] // 2
        sub_idx = spike_bins // 2
        return CamTargetSelection(
            activation=backbone_view,
            tensor_name="backbone_view",
            time_index=frame_idx * spike_bins + sub_idx,
            frame_contract=frame_contract,
            main_from_exec_rule=main_from_exec_rule,
            spike_bins=spike_bins,
        )

    return CamTargetSelection(
        activation=fused_main,
        tensor_name="fused_main",
        time_index=fused_main.shape[1] // 2 if fused_main.ndim == 5 else None,
        frame_contract=frame_contract,
        main_from_exec_rule=main_from_exec_rule,
        spike_bins=spike_bins,
    )


def compute_cam_map(
    activation: torch.Tensor,
    target: torch.Tensor,
    method: str,
    time_index: int | None,
) -> torch.Tensor:
    normalized = str(method).strip().lower()
    if normalized == "gradcam":
        return gradcam_from_activation(activation, target, time_index=time_index)
    if normalized == "hirescam":
        return hirescam_from_activation(activation, target, time_index=time_index)
    if normalized == "fallback":
        return gradcam_from_activation(activation, target, time_index=time_index)
    raise ValueError(f"Unsupported cam method: {method}")


def build_cam_metadata(
    *,
    requested_method: str,
    effective_method: str,
    default_scope: str,
    scopes_exported: list[str],
    selection: CamTargetSelection,
    analysis_crop_size: int,
    analysis_tile_stride: int,
    stitch_weight: str,
    roi_xyxy: tuple[int, int, int, int],
) -> dict:
    return {
        "cam_default_scope": default_scope,
        "cam_scopes_exported": scopes_exported,
        "cam_method_requested": requested_method,
        "cam_method_effective": effective_method,
        "cam_target_tensor": selection.tensor_name,
        "frame_contract": selection.frame_contract,
        "main_from_exec_rule": selection.main_from_exec_rule,
        "spike_bins": selection.spike_bins,
        "cam_time_index": selection.time_index,
        "analysis_crop_size": analysis_crop_size,
        "analysis_tile_stride": analysis_tile_stride,
        "analysis_tile_overlap": max(0, analysis_crop_size - analysis_tile_stride),
        "stitch_weight": stitch_weight,
        "roi_mask_xyxy": list(roi_xyxy),
    }
```

- [ ] **Step 2: Extend `maps.py` with explicit Grad-CAM / HiResCAM helpers**

Modify `scripts/analysis/fusion_attr/maps.py` to replace the single fallback helper with these functions:

```python
def _select_activation_slice(tensor: torch.Tensor, time_index: int | None) -> torch.Tensor:
    if tensor.ndim == 5:
        index = tensor.shape[1] // 2 if time_index is None else int(time_index)
        return tensor[0, index]
    if tensor.ndim == 4:
        return tensor[0]
    raise ValueError(f"Expected 4D or 5D activation, got {tuple(tensor.shape)}")


def gradcam_from_activation(activation: torch.Tensor, target: torch.Tensor, time_index: int | None = None) -> torch.Tensor:
    if activation.grad is not None:
        activation.grad.zero_()
    target.backward(retain_graph=True)
    if activation.grad is None:
        raise RuntimeError("Activation gradient was not retained")
    grad = _select_activation_slice(activation.grad.detach(), time_index)
    act = _select_activation_slice(activation.detach(), time_index)
    weights = grad.mean(dim=(-1, -2), keepdim=True)
    return torch.relu((weights * act).sum(dim=0))


def hirescam_from_activation(activation: torch.Tensor, target: torch.Tensor, time_index: int | None = None) -> torch.Tensor:
    if activation.grad is not None:
        activation.grad.zero_()
    target.backward(retain_graph=True)
    if activation.grad is None:
        raise RuntimeError("Activation gradient was not retained")
    grad = _select_activation_slice(activation.grad.detach(), time_index)
    act = _select_activation_slice(activation.detach(), time_index)
    return torch.relu((grad * act).sum(dim=0))


def gradient_activation_cam(activation: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return gradcam_from_activation(activation, target, time_index=None)
```

- [ ] **Step 3: Preserve structured probe outputs**

Modify `scripts/analysis/fusion_attr/probes.py` so the record keeps both the selected tensor and the structured dict:

```python
@dataclass
class FusionProbeRecord:
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor
    module_name: str
    structured_output: dict[str, Any] | None = None


def _hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
    tensor_output = _extract_tensor_output(output)
    if tensor_output is None:
        return
    if tensor_output.requires_grad:
        tensor_output.retain_grad()
    structured_output = output if isinstance(output, dict) else None
    if structured_output is not None:
        for value in structured_output.values():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                value.retain_grad()
    tensor_inputs = tuple(v for v in _detach_tensor(inputs) if isinstance(v, torch.Tensor))
    self.record = FusionProbeRecord(
        inputs=tensor_inputs,
        output=tensor_output,
        module_name=module.__class__.__name__,
        structured_output=structured_output,
    )
```

- [ ] **Step 4: Run the focused repair tests**

Run:

```bash
pytest tests/analysis/test_fusion_attr_cam_repair.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the CAM helper implementation**

```bash
git add scripts/analysis/fusion_attr/cam.py scripts/analysis/fusion_attr/maps.py scripts/analysis/fusion_attr/probes.py tests/analysis/test_fusion_attr_cam_repair.py
git commit -m "feat(analysis): add explicit cam target selection"
```

### Task 3: Implement Overlap-Weighted Tile Stitching And ROI Tile Helpers

**Files:**
- Create: `scripts/analysis/fusion_attr/stitching.py`
- Modify: `tests/analysis/test_fusion_attr_cam_repair.py`

- [ ] **Step 1: Implement the stitching helper module**

Create `scripts/analysis/fusion_attr/stitching.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TileBox:
    top: int
    left: int
    bottom: int
    right: int


def crop_box_to_tile(xyxy: tuple[int, int, int, int], tile: TileBox) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = xyxy
    local_x1 = max(x1, tile.left) - tile.left
    local_y1 = max(y1, tile.top) - tile.top
    local_x2 = min(x2, tile.right) - tile.left
    local_y2 = min(y2, tile.bottom) - tile.top
    if local_x2 <= local_x1 or local_y2 <= local_y1:
        return None
    return (local_x1, local_y1, local_x2, local_y2)


def mask_intersects_tile(xyxy: tuple[int, int, int, int], tile: TileBox) -> bool:
    return crop_box_to_tile(xyxy, tile) is not None


def build_hann_window(height: int, width: int) -> torch.Tensor:
    win_h = torch.hann_window(height, periodic=False)
    win_w = torch.hann_window(width, periodic=False)
    return torch.outer(win_h, win_w).clamp_min(1e-6)


def stitch_weighted_tiles(
    canvas_shape: tuple[int, int, int, int],
    tiles: list[tuple[torch.Tensor, TileBox]],
) -> torch.Tensor:
    accum = torch.zeros(canvas_shape, dtype=tiles[0][0].dtype)
    weight = torch.zeros(canvas_shape, dtype=tiles[0][0].dtype)
    for tile_tensor, tile in tiles:
        window = build_hann_window(tile_tensor.shape[-2], tile_tensor.shape[-1]).view(
            1, 1, tile_tensor.shape[-2], tile_tensor.shape[-1]
        )
        accum[:, :, tile.top:tile.bottom, tile.left:tile.right] += tile_tensor * window
        weight[:, :, tile.top:tile.bottom, tile.left:tile.right] += window
    return accum / weight.clamp_min(1e-6)
```

- [ ] **Step 2: Re-run the repair tests**

Run:

```bash
pytest tests/analysis/test_fusion_attr_cam_repair.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit the stitching helpers**

```bash
git add scripts/analysis/fusion_attr/stitching.py tests/analysis/test_fusion_attr_cam_repair.py
git commit -m "feat(analysis): add weighted tile stitching helpers"
```

### Task 4: Integrate Repaired CAM Flow Into The CLI And Export Contract

**Files:**
- Modify: `scripts/analysis/fusion_attribution.py`
- Modify: `tests/analysis/test_fusion_attr_panels_cli.py`
- Test: `tests/analysis/test_fusion_attr_cam_repair.py`
- Test: `tests/analysis/test_fusion_attr_panels_cli.py`

- [ ] **Step 1: Add CLI defaults and imports for the repaired flow**

Update the parser and imports in `scripts/analysis/fusion_attribution.py`:

```python
from scripts.analysis.fusion_attr.cam import build_cam_metadata, build_cam_scope_targets, compute_cam_map, select_cam_target
from scripts.analysis.fusion_attr.stitching import TileBox, crop_box_to_tile, mask_intersects_tile

parser.add_argument("--analysis-tile-stride", type=int, default=None)
parser.add_argument("--cam-scopes", nargs="+", default=["fullframe", "roi"], choices=["fullframe", "roi"])
parser.add_argument("--stitch-weight", default="hann", choices=["hann"])
```

and set:

```python
tile_stride = int(args.analysis_tile_stride or max(1, tile_size // 2))
```

- [ ] **Step 2: Replace hard-stitched single-CAM accumulation with scope-aware weighted stitching**

Replace the current `cam_accum` logic with explicit per-scope collections:

```python
cam_tiles: dict[str, list[tuple[torch.Tensor, TileBox]]] = {"fullframe": [], "roi": []}
selection_meta = None

for top, left, bottom, right in tiles:
    tile_box = TileBox(top=top, left=left, bottom=bottom, right=right)
    structured = probe.record.structured_output or {"fused_main": probe.record.output, "backbone_view": probe.record.output, "meta": {}}
    selection = select_cam_target(structured)
    if selection_meta is None:
        selection_meta = selection
    local_roi_xyxy = crop_box_to_tile(sample.xyxy, tile_box)
    scope_targets = build_cam_scope_targets(output=center_output, gt=center_gt, roi_xyxy=local_roi_xyxy)

    for scope_name, scope_target in scope_targets.items():
        if scope_name == "roi" and not mask_intersects_tile(sample.xyxy, tile_box):
            continue
        cam_tile = compute_cam_map(
            activation=selection.activation,
            target=scope_target,
            method=args.cam_method,
            time_index=selection.time_index,
        ).detach().cpu().unsqueeze(0).unsqueeze(0)
        cam_tiles[scope_name].append((cam_tile, tile_box))
```

then stitch after the loop:

```python
stitched_cams = {
    scope: stitch_weighted_tiles(
        canvas_shape=(1, 1, height, width),
        tiles=scope_tiles,
    )
    for scope, scope_tiles in cam_tiles.items()
    if scope_tiles
}
```

- [ ] **Step 3: Rename exported CAM files and expand metadata**

Replace the legacy single-CAM export block with:

```python
for scope_name, cam_map in stitched_cams.items():
    stem = f"cam_{scope_name}"
    np.save(str(maps_dir / f"{stem}_raw.npy"), cam_map[0, 0].numpy())
    save_gray_map_png(maps_dir / f"{stem}_gray.png", normalize_map(cam_map[0, 0]))
    cam_color = _save_color_map(maps_dir / f"{stem}_color.png", cam_map[0, 0])
    _save_overlay(overlays_dir / f"{stem}_on_blurry.png", blurry_bgr, cam_color)
    _save_overlay(overlays_dir / f"{stem}_on_restored.png", restored_bgr, cam_color)

metadata.update(
    build_cam_metadata(
        requested_method=args.cam_method,
        effective_method=args.cam_method,
        default_scope="fullframe",
        scopes_exported=sorted(stitched_cams.keys()),
        selection=selection_meta,
        analysis_crop_size=tile_size,
        analysis_tile_stride=tile_stride,
        stitch_weight=args.stitch_weight,
        roi_xyxy=sample.xyxy,
    )
)
metadata["cam_target_module"] = probe.record.module_name
```

- [ ] **Step 4: Update the CLI/export tests to the new contract**

Append these tests in `tests/analysis/test_fusion_attr_panels_cli.py`:

```python
def test_fusion_attribution_cli_help_mentions_cam_scopes():
    result = subprocess.run(
        [sys.executable, "scripts/analysis/fusion_attribution.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--cam-scopes" in result.stdout
    assert "--stitch-weight" in result.stdout


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
            "--cam-method",
            "hirescam",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cam_method"] == "hirescam"
```

- [ ] **Step 5: Run the analysis test suite**

Run:

```bash
pytest tests/analysis/test_fusion_attr_cam_repair.py tests/analysis/test_fusion_attr_panels_cli.py tests/analysis/test_fusion_attr_targets_maps.py tests/analysis/test_fusion_attr_probes.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the CLI integration**

```bash
git add scripts/analysis/fusion_attribution.py tests/analysis/test_fusion_attr_panels_cli.py tests/analysis/test_fusion_attr_cam_repair.py
git commit -m "fix(analysis): repair cam export semantics and stitching"
```

### Task 5: Add CLI Coverage For Repaired Metadata Fields

**Files:**
- Modify: `tests/analysis/test_fusion_attr_panels_cli.py`

- [ ] **Step 1: Extend the dry-run CLI test with repaired metadata assertions**

Update `test_fusion_attribution_cli_dry_run_manifest_keeps_requested_cam_method` so it also checks the manifest fields that remain available in dry-run mode:

```python
assert manifest["target"] == "masked_charbonnier"
assert manifest["cam_method"] in {"gradcam", "hirescam", "fallback"}
assert manifest["num_samples"] == 1
```

- [ ] **Step 2: Run the CLI tests**

Run:

```bash
pytest tests/analysis/test_fusion_attr_panels_cli.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit the export verification**

```bash
git add tests/analysis/test_fusion_attr_panels_cli.py
git commit -m "test(analysis): cover repaired cam dry-run metadata"
```

### Task 6: Final Verification

**Files:**
- Verify only

- [ ] **Step 1: Run the full attribution-related test set**

Run:

```bash
pytest tests/analysis/test_fusion_attr_cam_repair.py tests/analysis/test_fusion_attr_panels_cli.py tests/analysis/test_fusion_attr_targets_maps.py tests/analysis/test_fusion_attr_probes.py -q
```

Expected: PASS.

- [ ] **Step 2: Sanity-check the repaired CLI help**

Run:

```bash
python scripts/analysis/fusion_attribution.py --help
```

Expected: output includes `--cam-method`, `--cam-scopes`, `--stitch-weight`, `--analysis-tile-stride`.

- [ ] **Step 3: Commit the final verification checkpoint**

```bash
git commit --allow-empty -m "chore(analysis): verify repaired cam pipeline"
```
