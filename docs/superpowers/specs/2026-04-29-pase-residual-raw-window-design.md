# PASE-Residual Raw-Window Spike Design

**Date:** 2026-04-29

## Summary

Extend the existing `fusion.operator="pase_residual"` path so it can consume a per-frame raw spike window instead of only TFP bins.

The design keeps the current operator abstraction intact:

- operator name remains `pase_residual`
- structured early fusion contract remains `[B, T, S, H, W]`
- residual write-back logic remains unchanged

The main change is at the spike representation layer:

- current path: `.dat -> spike_matrix[T,H,W] -> TFP bins[S,H,W]`
- new path: `.dat -> spike_matrix[T,H,W] -> raw centered window[L,H,W]`

This lets `PASE` consume a true raw spike time window while keeping the comparison against TFP fair and localized.

## Problem

The current `pase_residual` design already fixes one major issue in the old `pase` fusion path: it lets `PASE` consume the full per-frame spike clip dimension `S` instead of processing one subframe at a time.

However, the spike clip that reaches `PASE` is still a TFP-derived representation:

- each RGB frame loads a corresponding `.dat`
- dataset code reconstructs that raw spike stream into `S` TFP bins
- `pase_residual` sees those bins as `[B, T, S, H, W]`

This means the current baseline still answers:

- how well can `PASE` fuse TFP-compressed spike evidence?

It does not answer:

- how well can `PASE` fuse a raw local spike time window before TFP compression?

If `PASE` is meant to exploit temporal structure directly, that second question is important.

## Goals

1. Keep `pase_residual` as the only operator touched in phase 1.
2. Let `PASE` consume a per-frame raw spike window shaped like `[L, H, W]`.
3. Preserve the existing structured early-fusion contract:
   - input: `rgb[B, T, 3, H, W]`, `spike[B, T, S, H, W]`
   - output: `fused[B, T, 3, H, W]`
4. Keep the change localized to dataset / spike representation handling as much as possible.
5. Preserve clean ablations between:
   - `pase_residual + tfp`
   - `pase_residual + raw_window`
6. Reuse the current TFP center-alignment semantics so the comparison changes representation, not time-reference rules.

## Non-Goals

1. Do not redesign `PixelAdaptiveSpikeEncoder`.
2. Do not add raw-window support to `fusion.operator="pase"`, `gated`, `mamba`, or `attention` in this phase.
3. Do not replace the existing TFP path.
4. Do not introduce a new operator name such as `pase_residual_raw`.
5. Do not silently auto-pad, truncate, or reinterpret invalid raw-window configurations.

## Design Decision

Implement raw-spike support as a representation-mode extension under the existing `pase_residual` operator, not as a new operator.

Rationale:

- the fusion architecture stays the same
- the spike representation becomes the controlled variable
- checkpoint, config, and ablation semantics stay cleaner

This keeps the comparison focused:

- same RGB encoder
- same PASE block
- same fusion body
- same write-back head
- different spike input representation

## Current Behavior

The present dataset pipeline does the following for each RGB neighbor frame:

1. load `spike/{neighbor}.dat`
2. decode it to `spike_matrix[T, H, W]`
3. reconstruct it into:
   - `TFP bins[S,H,W]`, or
   - `middle_tfp[1,H,W]`, or
   - `snn[1,H,W]`
4. crop and resize the resulting spike tensor to the RGB crop region
5. return dual inputs where spike has shape `[T_rgb, S, H, W]`

Under `pase_residual`, this becomes:

- `rgb[B, T, 3, H, W]`
- `spike[B, T, S, H, W]`
- `PASE(in_chans=S)`

This is already better than the old expanded single-channel path, but it still uses TFP-compressed bins.

## Proposed Representation Model

Add a spike representation mode that is resolved in the dataset:

- `representation="tfp"`: existing behavior
- `representation="raw_window"`: new behavior

Both modes still return a per-frame spike tensor shaped like `[S, H, W]`, but `S` means different things:

- in `tfp`, `S = number of TFP bins`
- in `raw_window`, `S = raw_window_length`

`pase_residual` itself should remain source-agnostic. It only consumes `[B, T, S, H, W]`.

## Time Alignment Semantics

The raw-window mode should reuse the same center semantics as the current TFP path.

That means:

- each RGB frame still maps to its paired spike clip file
- the dataset still treats that clip as the local temporal evidence for that RGB frame
- the default raw window is centered on the clip's default center, i.e. the same temporal reference that current TFP reconstruction uses

Phase 1 should not introduce a raw-specific center rule.

Reason:

- it keeps TFP vs raw-window comparisons fair
- it avoids multiplying configuration complexity
- it matches the original PASE intuition better than inventing a second alignment convention

## Raw Window Length

The raw-window mode needs an explicit temporal width parameter because `PASE` consumes the entire input channel depth it is given.

Use:

- `raw_window_length: null | int`

Rules:

1. if `null`, resolve to `2 * tfp_half_win_length + 1`
2. if explicitly set, use that value
3. value must be a positive odd integer

Rationale:

- default behavior stays aligned with the current TFP temporal coverage
- experiments can still sweep the raw window size independently
- config files can clearly indicate whether they follow the derived default or intentionally override it

## Dataset Behavior

### 1. Decode Once Per Frame

For each frame-level spike file:

- decode `.dat` to `spike_matrix[T, H, W]`

This already exists today and should remain the shared starting point for both modes.

### 2. Representation Branch

If `representation="tfp"`:

- keep existing `voxelize_spikes_tfp(...) -> [S, H, W]`

If `representation="raw_window"`:

- resolve effective `L = raw_window_length`
- take a centered temporal crop from `spike_matrix`
- return `raw_window[L, H, W]`

The centered crop should be symmetric around the default center so long as the chosen `L` allows it.

### 3. Spatial Crop / Resize

After the representation tensor is built:

- keep the current crop-to-RGB-region behavior
- keep the current resize-to-RGB-crop-size behavior

This ensures raw-window and TFP inputs experience the same spatial pipeline.

### 4. Dual Tensor Contract

The dual-input contract should remain:

- `L_rgb[T, 3, H, W]`
- `L_spike[T, S, H, W]`

Only the meaning of `S` changes under `raw_window`.

## Operator Behavior

`PaseResidualFusionOperator` should not branch on representation type.

It should continue to:

1. receive `spike[B, T, S, H, W]`
2. reshape to `spike_flat[B*T, S, H, W]`
3. run `PixelAdaptiveSpikeEncoder(in_chans=S, out_chans=C, ...)`
4. fuse with RGB context and write back a conservative residual

This keeps the fusion architecture stable and makes the spike representation the primary experimental variable.

## Configuration Model

Phase 1 should preserve the current operator name and add representation configuration at the data/input layer.

Recommended shape:

```json
"spike": {
  "representation": "raw_window",
  "reconstruction": {
    "type": "spikecv_tfp",
    "num_bins": 4
  },
  "raw_window_length": null
}
```

Meaning:

- `representation` selects what is fed into fusion
- `reconstruction.type` remains relevant for TFP mode and legacy compatibility
- `raw_window_length` is only used when `representation="raw_window"`

Implementation may support legacy config aliases, but the canonical semantics should be explicit.

## Failure Strategy

This design should fail fast instead of silently correcting invalid settings.

Reject cases such as:

1. `raw_window_length <= 0`
2. `raw_window_length` is even
3. effective `raw_window_length > available T`
4. `representation="raw_window"` with a fusion operator other than `pase_residual` in phase 1
5. inconsistent train/test representation settings when the run expects parity

Avoid these behaviors:

- silent zero-padding
- silent truncation
- silently falling back from `raw_window` to `tfp`

The user should know exactly which representation was used.

## Metadata and Logging

Add lightweight metadata so experiments and analysis scripts can identify the active spike representation.

Recommended metadata keys:

- `spike_representation`
- `spike_window_length`
- `effective_spike_channels`

These should flow through the existing fusion metadata / diagnostics path where practical.

This is especially useful for:

- training log interpretation
- ablation result tables
- offline attribution / fusion analysis scripts

## Testing Strategy

### Dataset Tests

Add tests that verify:

1. `representation="tfp"` preserves current behavior
2. `representation="raw_window"` returns centered `[L,H,W]` windows
3. `raw_window_length=null` resolves to `2 * tfp_half_win_length + 1`
4. invalid even / negative / overlong window sizes raise clear errors

### Contract Tests

Add tests that verify:

1. train and test datasets both return `L_spike[T,S,H,W]` under `raw_window`
2. crop / resize keeps spike spatial size aligned with RGB
3. dual-input contract remains valid

### Fusion Integration Tests

Add tests that verify:

1. VRT builds with `fusion.operator="pase_residual"` and `representation="raw_window"`
2. `pase_residual` forward shape is unchanged
3. using `raw_window` with non-`pase_residual` fusion rejects configuration in phase 1

### Regression Tests

Keep or add tests that confirm:

1. existing `pase_residual + tfp` configs still build
2. existing TFP-only training paths still behave as before

## Experimental Plan Guidance

For the first controlled study:

- baseline: `pase_residual + tfp`
- variant: `pase_residual + raw_window`

Keep fixed:

- same operator
- same crop pipeline
- same training loop
- same loss setup
- same warmup behavior

Default fair comparison:

- `tfp_half_win_length = 20`
- derived `raw_window_length = 41`

Then optionally sweep:

- `L = 21`
- `L = 41`
- `L = 61`

This isolates whether gains come from:

- the raw representation itself
- or simply a different temporal support size

## Acceptance Criteria

The design is successful if:

1. `pase_residual` can consume raw per-frame spike windows without changing its external contract
2. the default raw window stays temporally aligned with the current TFP path
3. TFP and raw-window modes can be compared under the same fusion architecture
4. invalid raw-window configs fail clearly
5. existing TFP behavior remains available for regression and ablation

## Risks and Mitigations

### Risk: representation semantics leak into operator logic

Mitigation:

- keep branching in dataset / input preparation
- keep `pase_residual` source-agnostic

### Risk: raw-window length accidentally diverges from the intended fair TFP comparison

Mitigation:

- derived default from `tfp_half_win_length`
- log resolved effective window length

### Risk: train/test configuration drift causes misleading experiments

Mitigation:

- validate representation settings explicitly
- surface them in logs and manifests

### Risk: raw-window inputs increase memory or runtime enough to distort comparisons

Mitigation:

- log effective spike channel count
- record runtime and memory alongside quality metrics
- treat window-length sweeps as explicit ablations, not hidden defaults

## Implementation Scope

Primary files expected to change:

- `data/dataset_video_train_rgbspike.py`
- `data/dataset_video_test.py`
- `models/architectures/vrt/vrt.py`
- `tests/data/...` for dataset representation tests
- `tests/models/test_vrt_fusion_integration.py`
- `tests/models/test_fusion_early_adapter.py` if representation-gating checks live there
- `options/...` for at least one raw-window `pase_residual` config

This remains a single focused implementation unit and is ready to be turned into an implementation plan after review.
