# Early Fusion Mamba Redesign Design

## Summary

This design replaces the current early-fusion `mamba` operator with an early-only, RGB-conditioned spike-sequence fusion path.
The new operator consumes spike bin-level temporal information inside fusion, outputs one RGB-aligned fused frame per original RGB frame, and leaves frame-to-frame temporal modeling to the VRT backbone.

## Problem

The current `mamba` early-fusion implementation is not semantically aligned with the project's early fusion goal:

- The early adapter expands `[B, T, S, H, W]` spike input into a flattened `[B, T*S, 1, H, W]` sequence and repeats RGB per spike bin.
- The current `mamba` operator then models that flattened `T*S` axis as a single sequence.
- This mixes frame time and spike-bin time into one undifferentiated axis.
- The current operator also lacks a conservative residual initialization path, so it can perturb RGB aggressively at initialization.

This makes the existing design hard to interpret, poorly matched to the intended supervision target, and risky to train.

## Goals

- Consume spike bin-level temporal information inside early fusion.
- Produce one fused RGB-aligned output per original RGB frame.
- Keep VRT responsible for video frame temporal modeling only.
- Preserve stable optimization through residual correction and conservative initialization.
- Make the `mamba` operator explicitly early-only instead of pretending it shares the same semantics as all legacy fusion operators.

## Non-Goals

- Do not redesign `middle` or `hybrid` fusion.
- Do not make `mamba` a generic feature fusion operator for arbitrary channel counts.
- Do not preserve compatibility with old bin-expanded early semantics for `mamba`.
- Do not introduce staged rollout variants; implement the final intended behavior directly.

## Design Decision

### Temporal responsibility split

The design adopts a two-level temporal split:

- Fusion models spike bin-level temporal information within each RGB frame.
- VRT models frame-to-frame temporal information across the video sequence.

This means spike micro-temporal structure is compressed into a frame-level fused representation before entering VRT.

### Output contract

For `fusion.placement="early"` and `fusion.operator="mamba"`:

- Input RGB shape: `[B, T, 3, H, W]`
- Input spike shape: `[B, T, S, H, W]`
- Output fused shape: `[B, T, 3, H, W]`

The `mamba` operator no longer participates in the old early-fusion contract where outputs were expanded to `T*S` timesteps.

## Architecture

### Early adapter behavior

The early adapter keeps the existing path for legacy operators:

- `concat`
- `gated`
- `pase`

For `mamba`, the early adapter switches to a structured path:

- It does not repeat RGB per spike bin.
- It does not flatten `T` and `S` into one axis.
- It forwards structured tensors directly to the `mamba` operator.

### Mamba operator structure

The redesigned `MambaFusionOperator` is an RGB-conditioned spike-sequence fusion module with these submodules:

1. `rgb_encoder`
   - Lightweight 2D convolutional stack.
   - Maps `3 -> D`.
   - Produces per-frame conditioning features `rgb_ctx`.

2. `spike_token_proj`
   - Shared projection from `1 -> D` for each spike bin.
   - Produces spike token features.

3. `conditioning`
   - Default mode is additive conditioning.
   - Each spike token is conditioned by the corresponding frame's RGB context.
   - Initial implementation supports `rgb_cond="add"`.

4. `mamba_stack`
   - Runs only along the spike-bin axis `S`.
   - Uses `num_layers=3` by default.
   - Each layer uses Mamba with hidden width `D=48`.
   - Normalization is applied after each block.

5. `token_pool`
   - Aggregates along the spike-bin axis.
   - Default pooling is mean pooling.

6. `residual_head`
   - `correction_head`: predicts RGB correction.
   - `gate_head`: predicts a per-pixel gate.
   - Final output is `rgb + sigmoid(gate + bias_init) * correction`.

### Capacity defaults

Default `mamba` capacity is chosen to avoid the underpowered behavior already observed with overly small fusion networks:

- `model_dim=48`
- `d_state=32`
- `d_conv=4`
- `expand=2`
- `num_layers=3`

This is intentionally larger than a minimal toy fusion module, but still much smaller than the VRT backbone.

## Data Flow

For each frame:

1. Encode RGB frame into `rgb_ctx`.
2. Project each spike bin into a token embedding.
3. Add the corresponding RGB conditioning feature to each token.
4. Run the sequence of `S` spike tokens through the Mamba stack.
5. Pool the resulting token sequence over `S`.
6. Predict correction and gate from the pooled feature.
7. Add gated correction back onto the original RGB frame.

Across the batch and time dimensions:

- Frames are processed independently inside fusion.
- VRT later receives the fused frame sequence and performs video temporal modeling over `T`.

## Initialization and Optimization

The redesigned operator must start near RGB passthrough:

- The last layer of `correction_head` is zero-initialized.
- The final gate bias is initialized to `-5.0`.
- The residual output therefore starts near zero.
- The RGB path remains the dominant signal at initialization.

This matches the stabilization intent already used in the redesigned gated fusion operator.

## Configuration

The `mamba` operator becomes an explicitly constrained early-only path.

Required constraints:

- `fusion.placement == "early"`
- `fusion.operator == "mamba"`
- `fusion.out_chans == 3`
- `fusion.early.expand_to_full_t == false`

Default operator parameters:

```json
{
  "model_dim": 48,
  "d_state": 32,
  "d_conv": 4,
  "expand": 2,
  "num_layers": 3,
  "rgb_cond": "add",
  "token_pool": "mean",
  "init_gate_bias": -5.0
}
```

### Explicit incompatibilities

The code must reject invalid combinations instead of attempting implicit compatibility:

- `placement != "early"` with `operator="mamba"`
- `out_chans != 3` with `operator="mamba"`
- `early.expand_to_full_t == true` with `operator="mamba"`
- Any path that expects `mamba` early fusion to emit `T*S` outputs

## Code Changes

### `models/fusion/operators/mamba.py`

Replace the current flattened-sequence design with the structured, RGB-conditioned, residual early-fusion operator.

### `models/fusion/adapters/early.py`

Add a structured forward path for `operator="mamba"`:

- preserve current legacy behavior for `concat/gated/pase`
- bypass RGB repetition and `T*S` flattening for `mamba`

### `models/architectures/vrt/vrt.py`

Add validation for the new `mamba` constraints:

- early-only
- no full-T expansion
- no incompatible output expectations

Keep the VRT backbone input width logic consistent with frame-level fused RGB output.

### Tests

Add or update tests to cover:

- `mamba` early operator output shape `[B, T, 3, H, W]`
- conservative initialization behavior
- invalid config rejection
- early adapter structured `mamba` path
- VRT integration with valid `early+mamba` config
- runtime dependency handling when `mamba_ssm` is absent

## Tradeoffs

### Benefits

- Clear semantics: fusion models spike bin-level time, VRT models video frame time.
- Better training stability via residual design and conservative initialization.
- Better alignment between fusion output and RGB GT supervision.
- Avoids contaminating VRT with mixed temporal scales.

### Costs

- `mamba` no longer shares identical operator semantics with legacy early operators.
- Some previous early full-T behaviors become explicitly unsupported for `mamba`.
- The implementation is more specialized and therefore less generic.

## Alternatives Considered

### Keep flattened `T*S` sequence modeling

Rejected because frame time and spike-bin time are different temporal concepts and should not be mixed implicitly.

### Keep old output semantics and expand `mamba` output to `T*S`

Rejected because the design goal is RGB-aligned frame restoration, not generating one fused pseudo-frame per spike bin.

### Let VRT consume spike bin-level time directly

Rejected for this redesign because it would force the backbone to model mixed temporal scales and would substantially complicate semantics and compute.

## Acceptance Criteria

- `early+mamba` consumes structured `[B, T, 3, H, W]` and `[B, T, S, H, W]` inputs.
- `early+mamba` outputs `[B, T, 3, H, W]`.
- Invalid `mamba` configurations fail with explicit validation errors.
- Initialization is near RGB passthrough.
- The default capacity uses `model_dim=48`, `d_state=32`, `d_conv=4`, `expand=2`, `num_layers=3`.
- Existing non-`mamba` early operators keep their current behavior.

## Open Questions

No unresolved design questions remain for the initial implementation.
