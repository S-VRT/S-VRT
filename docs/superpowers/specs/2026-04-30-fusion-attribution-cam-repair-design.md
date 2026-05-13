# Fusion Attribution CAM Repair Design

**Date:** 2026-04-30

## Summary

Repair the current fusion attribution CAM pipeline so the exported maps have clear semantics, align with the actual fusion tensor consumed by the backbone, and no longer show strong tile-shaped stitching artifacts caused by hard patch composition.

The repaired pipeline should export two CAM scopes by default:

- full-frame CAM: explains which fusion-adjacent activations support the restoration target over the whole center output frame
- ROI CAM: explains which activations support restoration inside the user-provided `sample.mask`

This repair intentionally updates existing filenames and metadata semantics instead of preserving legacy ambiguity. The output contract should become explicit enough that a reader can tell:

- what target was used
- which fusion tensor was attributed
- how the center frame was aligned
- whether the map is full-frame or ROI-only
- how tiled stitching was performed

## Problem

The current CAM export path mixes three different issues:

1. incorrect target semantics in tiled mode
2. incomplete target-layer and time-axis alignment
3. visible block artifacts caused by hard tile stitching

The most important semantic bug is that tiled attribution currently replaces the sample ROI with the full tile extent before constructing the target mask. As a result:

- the exported CAM is not a true ROI CAM
- the map answers a different question than the samples file claims
- the metadata does not reveal this semantic drift clearly enough

The second issue is attribution alignment. The current probe captures the first available fusion-adjacent tensor and applies a simplified CAM reduction on top of it. That is acceptable only when the tensor time axis exactly matches the main restoration path. In expanded early fusion, that assumption is not guaranteed.

The third issue is visualization quality. Each tile is currently processed independently and stitched back with hard spatial boundaries. This produces obvious rectangular seams that are dominated by attribution implementation artifacts rather than model behavior.

## Goals

1. Make CAM output semantics explicit and correct.
2. Export full-frame CAM and ROI CAM in the same run by default.
3. Ensure attribution is aligned to the effective fusion tensor used by the main restoration path.
4. Remove visible tile-boundary artifacts under the default configuration.
5. Make `--cam-method` select a real implementation instead of acting as a metadata-only flag.
6. Update filenames and metadata so downstream analysis can interpret the new outputs unambiguously.
7. Add tests that cover scope semantics, time alignment, tiled stitching, and metadata export.

## Non-Goals

1. Do not redesign the training loop or move attribution into training.
2. Do not make tiled full-frame CAM mathematically identical to single-pass whole-image attribution in phase 1.
3. Do not refactor IG, PCA, perturbation maps, and CAM into a large new framework unless required by this repair.
4. Do not preserve old ambiguous filenames as the primary output contract.
5. Do not treat CAM as standalone evidence of restoration quality.

## Design Decision

Implement a medium-scope attribution repair with four coordinated changes:

1. explicit CAM scopes
2. explicit fusion target alignment
3. overlap-and-weight tile stitching
4. explicit metadata and filename migration

This is larger than a minimal bug fix, but smaller than a full attribution framework rewrite. It matches the need to fix semantics and remove block artifacts in the same pass without turning the task into a broad redesign.

## Current Behavior

The current pipeline can be summarized as follows:

1. split the input into spatial tiles
2. run model forward on each tile
3. probe the fusion adapter output
4. compute a scalar restoration target
5. backpropagate and build a simplified CAM map
6. write each tile result into a full-frame canvas
7. normalize the final map and save it as `cam.png` and `cam_raw.npy`

The main current failure modes are:

- ROI mask semantics are lost in tiled execution because the tile mask is expanded to the full tile.
- expanded early fusion may attribute the wrong temporal slice if the probe output uses execution-time subframes while the restoration path uses a reduced main view.
- tile maps are stitched with hard boundaries and uniform averaging.
- the `--cam-method` argument does not control the actual implementation used for CAM generation.
- output filenames do not distinguish between full-frame and ROI semantics.

## Scope Semantics

The repaired pipeline should export two CAM scopes by default.

### 1. Full-Frame CAM

Full-frame CAM answers:

> Which fusion-adjacent activations matter most for the restoration target over the whole center output frame?

This map should be saved as the default CAM scope for paper-style overview figures.

### 2. ROI CAM

ROI CAM answers:

> Which fusion-adjacent activations matter most for restoration inside the user-specified mask region?

This map should be exported in the same run whenever a sample mask is available.

### Scope Rules

- `fullframe` and `roi` are separate attribution scopes, not visualization variants of the same tensor.
- Each scope should produce its own raw map, grayscale image, color image, and overlays.
- The default scope should be `fullframe`.
- Metadata must declare both the default scope and the full exported scope list.

## Target Semantics

The restoration target should be separated by scope.

### Full-Frame Target

Use the center output frame and center GT frame over the whole spatial extent:

```text
target_fullframe = -Charbonnier(output_center_frame, gt_center_frame)
```

### ROI Target

Use the real sample mask without expanding it to tile extents:

```text
target_roi = -Charbonnier(output_center_frame * M, gt_center_frame * M)
```

### Tiled Scope Behavior

In tiled mode:

- full-frame CAM should compute a local target restricted to the tile support and stitch those local contributions into a full-frame approximation
- ROI CAM should crop the true ROI mask into tile coordinates
- tiles with no ROI overlap may skip ROI backward for efficiency and lower noise

Under no circumstances should the ROI target silently become a full-tile target.

## Target Layer and Time Alignment

CAM should be generated from the effective fusion tensor used by the main restoration path, not merely the first tensor available from a forward hook.

### Early Fusion With Collapsed Frame Contract

For collapsed early fusion:

- attribute `fused_main`
- use the center frame index on the main time axis

This covers operators such as:

- `pase_residual`
- `mamba`
- `attention`

### Early Fusion With Expanded Frame Contract

For expanded early fusion:

- do not treat the midpoint of `backbone_view` as the attribution frame by default
- align attribution to the `fused_main` semantics used by the backbone
- if attribution must originate from execution-time subframes, apply the same `main_from_exec_rule` used to reduce execution-time views to main-frame views

This is especially important when the operator emits `N * spike_bins` execution steps but the restoration path consumes only the center subframe per RGB frame.

### Middle and Hybrid Fusion

The pipeline should explicitly record which fusion module and which tensor were attributed:

- target module name
- target tensor name
- fusion placement

The first repair phase does not need to support multi-layer CAM aggregation. It only needs to make the chosen attribution target explicit and correct.

## CAM Method Dispatch

The `--cam-method` argument must control the implementation used.

Recommended supported methods:

- `gradcam`: standard gradient-weighted channel aggregation
- `hirescam`: element-wise `grad * activation` before channel reduction
- `fallback`: repo-local simplified implementation retained for compatibility and testing

Rules:

- metadata must record both requested and effective method
- unsupported method / tensor combinations must fail loudly instead of silently falling back
- `fallback` should be described as simplified, not as identical to external reference implementations

## Tile Stitching Repair

The tiled execution model should remain, but the stitching strategy must change.

### 1. Default Overlap

Change the default stride from:

```text
stride = tile_size
```

to:

```text
stride = tile_size // 2
```

This creates 50% overlap by default and reduces seam visibility substantially.

### 2. Spatial Weighting

Each tile contribution should be multiplied by a smooth spatial weight window before accumulation.

Recommended default:

- 2D Hann window

Acceptable alternative:

- Tukey window

The intent is the same:

- center pixels receive high weight
- tile edges receive low weight
- overlapping tiles blend smoothly after normalization

### 3. Stitching Rule

For each map type:

1. compute raw tile map
2. multiply by spatial window
3. accumulate weighted tile map
4. accumulate weights
5. divide by accumulated weights after all tiles are processed

Normalization for visualization should happen only once on the stitched full-frame result.

Per-tile normalization before stitching should not be used.

### 4. Scope-Aware Tile Execution

For full-frame CAM:

- every tile contributes according to its local target support

For ROI CAM:

- only tiles intersecting the ROI should contribute nonzero attribution

### 5. Legacy-Like Mode

If the user manually sets:

```text
analysis_tile_stride >= analysis_crop_size
```

metadata should clearly indicate that hard-stitch-like behavior was requested and that visible seams may remain.

## Output Contract

Legacy ambiguous CAM filenames should be replaced with scope-aware filenames.

Recommended outputs:

```text
maps/
  cam_fullframe_raw.npy
  cam_fullframe_gray.png
  cam_fullframe_color.png
  cam_roi_raw.npy
  cam_roi_gray.png
  cam_roi_color.png
overlays/
  cam_fullframe_on_blurry.png
  cam_fullframe_on_restored.png
  cam_roi_on_blurry.png
  cam_roi_on_restored.png
```

If fusion-specific maps such as `effective_update` are exported, they should remain separate from CAM scope naming and keep their own semantics.

## Metadata Contract

`metadata.json` should be expanded so each run is self-describing.

Required fields:

- `cam_default_scope`
- `cam_scopes_exported`
- `cam_method_requested`
- `cam_method_effective`
- `cam_target_module`
- `cam_target_tensor`
- `fusion_placement`
- `frame_contract`
- `main_from_exec_rule`
- `spike_bins`
- `cam_time_index`
- `analysis_crop_size`
- `analysis_tile_stride`
- `analysis_tile_overlap`
- `stitch_weight`
- `roi_mask_xyxy`

Desirable additional fields:

- whether ROI tiles with zero overlap were skipped
- tile count per scope
- whether tiled attribution is an approximation of whole-frame attribution

## Reading Guidance

The repaired outputs should support the following interpretation rules:

- full-frame CAM is for understanding global restoration dependence
- ROI CAM is for understanding local restoration dependence inside the selected mask
- dark regions are low positive contribution under the chosen method and target, not proof of irrelevance
- cross-sample visual comparison still depends on shared normalization and should be made carefully

The docs and figure captions should state that CAM localizes restoration-sensitive fusion regions, while quantitative metrics and error maps remain primary evidence.

## Testing Strategy

The repair should add coverage in three layers.

### 1. Unit Tests

Add tests for:

- ROI mask cropping into tile coordinates
- expanded early fusion center-subframe alignment
- CAM method dispatch
- weighted stitching behavior

### 2. Numerical / Behavioral Tests

Use controlled toy activations or synthetic tile maps to verify:

- overlap-and-weight stitching is smoother than hard stitching
- final stitched maps do not show step discontinuities under the same synthetic input
- ROI CAM is zero or skipped outside ROI support

### 3. CLI / Export Tests

Verify:

- new filenames are written
- metadata fields are present
- default CAM scope is `fullframe`
- both `fullframe` and `roi` outputs are exported for masked samples

## Migration

This repair intentionally updates the meaning of the default CAM output.

Rules:

- old `cam.png`, `cam_color.png`, and `cam_raw.npy` should no longer be the primary contract
- the new outputs should replace them with explicit scope names
- docs should state that CAM exports generated after 2026-04-30 are not semantically identical to older ambiguous CAM outputs

This is acceptable because the current request explicitly allows replacing old names and fields as part of the repair.

## Risks

1. tiled full-frame CAM remains an approximation and may still differ from a true whole-frame backward pass
2. expanded early fusion alignment may expose hidden assumptions in the current probe design
3. changing output names may require updating downstream scripts or paper-figure assembly code
4. overlap-and-weight stitching increases compute relative to non-overlapping tiles

These are acceptable trade-offs because the current outputs are semantically misleading and visually contaminated by implementation artifacts.

## Recommended Implementation Order

1. fix scope semantics and target construction
2. fix target-tensor / time-axis alignment
3. implement CAM method dispatch
4. implement overlap-and-weight stitching
5. migrate filenames and metadata
6. add tests and export verification

## Success Criteria

The repair is successful when all of the following are true:

1. default CAM output clearly represents full-frame attribution
2. ROI CAM is exported separately and uses the true sample mask
3. expanded and collapsed early fusion both attribute the correct center-frame fusion signal
4. `--cam-method` changes the implementation actually used
5. default outputs no longer show obvious rectangular tile seams
6. metadata is sufficient to explain exactly what each CAM file represents
