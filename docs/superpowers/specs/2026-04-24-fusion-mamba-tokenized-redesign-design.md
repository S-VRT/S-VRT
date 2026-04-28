# Fusion Mamba Tokenized Redesign Design

**Date:** 2026-04-24

## Summary

Redesign early-fusion `mamba` as a tokenized, RGB-conditioned spike fusion operator that keeps the existing collapsed early-fusion contract while replacing the current per-pixel spike-bin sequence path.

The new design keeps `mamba` in the system, but changes three things together:

- input semantics: from per-pixel spike-bin sequences to low-resolution spike-aware tokens
- initialization: from near-closed gradient paths to small-output, non-degenerate write-back
- training: from a single phase-1 fusion objective to staged fusion warmup

The goal is to preserve temporal modeling inside fusion without repeating the current failure mode where phase-1 loss oscillates for thousands of iterations with little evidence of useful fusion learning.

## Problem

The current `mamba` fusion path is not failing because “Mamba is bad.” It is failing because the present operator semantics, initialization, and phase-1 training path work against each other.

Observed context from the current code path:

- the active training config keeps the run in phase 1 until `fix_iter=20000`
- in phase 1, `ModelPlain._phase1_fusion_forward()` trains only the fusion path and does not run the full VRT backbone
- the current `MambaFusionOperator` works on per-pixel spike-bin sequences with output
  `rgb + sigmoid(gate) * correction`
- `correction_head[-1]` is zero-initialized
- `gate_head[-1].weight` is zero-initialized
- `gate_head[-1].bias` defaults to `-5.0`

This creates an overly conservative startup regime:

- initial effective update is almost zero
- early gradients are dominated by the last correction layer
- upstream RGB encoder, spike projection, and Mamba blocks receive extremely weak learning signal
- phase-1 supervision only sees final RGB reconstruction error, which is too distant to efficiently bootstrap the current operator

The result is a fusion module that appears “stable” at initialization but is practically hard to start learning.

## Root-Cause Diagnosis

The main root cause is not one isolated bug. It is a design mismatch across four layers:

1. **Operator granularity**
   - current Mamba runs on an enormous number of independent per-pixel spike-bin sequences
   - this is expensive and hard to optimize for a module that only needs to produce a conservative RGB correction

2. **Conditioning strategy**
   - RGB context is added directly to spike token embeddings before sequence modeling
   - this is simple, but it does not create a strong, structured path for spike-aware conditional fusion

3. **Initialization**
   - output safety is currently achieved by nearly shutting down the write-back path
   - this suppresses gradients instead of merely suppressing output magnitude

4. **Phase-1 optimization**
   - phase 1 asks the fusion module to solve a final pixel-space objective immediately
   - the hardest components of the operator are active from the start
   - the optimization path is too long for such a weakly initialized module

## Goals

1. Keep `mamba` as the temporal modeling core of early fusion.
2. Preserve the current collapsed early-fusion contract:
   - input: `rgb[B,T,3,H,W]`, `spike[B,T,S,H,W]`
   - output: `fused[B,T,3,H,W]`
3. Move Mamba computation from full-resolution per-pixel sequences to low-resolution spike-aware tokens.
4. Keep the fusion write-back path conservative without collapsing gradients.
5. Introduce a staged phase-1 warmup that lets the operator learn to write safe RGB corrections before fully training the temporal mixer.
6. Add explicit fusion diagnostics so future failures can be diagnosed from evidence rather than loss shape alone.

## Non-Goals

1. Do not redesign the existing expanded-contract operators (`gated`, `concat`, `pase`) in this change.
2. Do not change the outer early-fusion wrapper contract for `mamba`.
3. Do not shift frame-to-frame temporal modeling responsibility away from VRT.
4. Do not redesign middle or hybrid fusion.
5. Do not require dataset-side changes to the current dual-input payload format.

## Design Decision

Adopt a **tokenized frame-local Mamba fusion** design:

- each RGB frame produces low-resolution context features
- each spike clip produces a small set of spike-aware spatial tokens
- Mamba models spike-bin temporal structure on those tokens
- a lightweight write-back head converts the token summary into a small RGB residual

This retains the original intent of using Mamba to capture micro-temporal spike structure while removing the most optimization-hostile part of the current design: per-pixel sequence modeling under nearly closed residual gates.

## Architecture

### 1. RGB Context Encoder

Input:

- `rgb[B,T,3,H,W]`

Output:

- `rgb_ctx[B,T,D,H',W']`

Responsibilities:

- provide stable RGB-conditioned context for fusion
- reduce spatial resolution before token mixing
- avoid heavy computation; this should be a lightweight convolutional stack

Recommended default behavior:

- two small conv blocks
- spatial downsampling by stride or patching to `H' x W'`
- output width `D` aligned with token dimension

### 2. Spike Token Encoder

Input:

- `spike[B,T,S,H,W]`

Output:

- `spike_tokens[B,T,N,S,D]`

Responsibilities:

- convert each frame’s spike bins into low-resolution token sequences
- reduce spatial complexity from `H*W` pixels to `N` tokens
- preserve spike-bin axis `S` so Mamba can model micro-temporal structure explicitly

Recommended default behavior:

- spatial patching or strided projection
- token dimension `D` shared with RGB context
- token count `N` derived from low-resolution spatial grid instead of raw pixels

### 3. Conditional Mamba Token Mixer

Input:

- spike-derived token sequences conditioned by RGB context

Output:

- temporally mixed token summaries per frame

Responsibilities:

- model spike-bin structure within each RGB frame
- keep frame-wise semantics explicit
- avoid entangling spike-bin time with video frame time

Recommended behavior:

- for each frame and token location, run Mamba along the `S` axis only
- inject RGB context as conditioning to each token sequence before or during the Mamba stack
- pool over `S` after temporal mixing to get one fused token per frame and token location

This keeps temporal responsibility split clean:

- fusion models spike-bin micro-time
- VRT models video frame time

### 4. Fusion Write-Back Head

Input:

- fused token map at low spatial resolution

Output:

- `delta_rgb[B,T,3,H,W]`
- `gate[B,T,3,H,W]`
- final fused output `rgb + alpha * sigmoid(gate) * delta_rgb`

Responsibilities:

- turn token-level fusion summary into a safe RGB correction
- keep initial updates small but learnable
- provide interpretable control over effective update magnitude

Key design rule:

- introduce an explicit residual scaling parameter `alpha`
- use `alpha` to control startup magnitude instead of using a nearly closed gate to suppress learning

### 5. Diagnostics Hook

The operator should expose stable diagnostics after forward:

- `token_norm`
- `mamba_norm`
- `delta_norm`
- `gate_mean`
- `effective_update_norm`
- `warmup_stage` when available from training path

These diagnostics should be treated as first-class fusion metadata, not ad hoc debug-only tensors.

## Input and Output Semantics

The external operator contract remains unchanged:

- input RGB: `rgb[B,T,3,H,W]`
- input spike: `spike[B,T,S,H,W]`
- output fused RGB: `[B,T,3,H,W]`

The new semantics are internal:

- Mamba no longer consumes one spike-bin sequence per original image pixel
- Mamba consumes one spike-bin sequence per low-resolution token
- write-back happens after token mixing, not before

This preserves compatibility with the current collapsed early-fusion wrapper while significantly changing the optimization geometry inside the operator.

## Initialization Strategy

The current “safe initialization” is too aggressive. The redesign uses **small-output, non-degenerate initialization** instead.

### New initialization rules

1. `delta_rgb` head final layer:
   - small random initialization
   - example scale: `std ~= 1e-3`
   - do not zero it completely

2. `gate` head final layer:
   - small random initialization
   - mild negative bias
   - recommended initial bias around `-2.0`, not `-5.0`

3. `alpha` residual scale:
   - explicit scalar or tiny per-channel parameter
   - recommended initial value in the range `0.05` to `0.1`

### Rationale

This preserves the desired startup behavior:

- initial RGB output is still close to passthrough
- effective update is still small

But it avoids the current failure mode:

- write-back path is not effectively closed
- upstream encoder and Mamba stack receive meaningful gradients from the beginning

## Training Redesign

## Phase-1 Warmup Strategy

Replace the current single-behavior phase 1 with staged warmup inside the existing phase-1 window.

### Stage 1: Write-Back Warmup

Train only:

- fusion write-back head
- `alpha`

Freeze:

- Mamba stack
- RGB context encoder
- spike token encoder
- VRT backbone

Purpose:

- teach the operator to produce safe, directed RGB updates
- establish a descending loss signal before enabling the hardest temporal components

### Stage 2: Token Fusion Warmup

Unfreeze:

- spike token encoder
- Mamba stack

Keep VRT backbone frozen.

Purpose:

- once the write-back path is already learnable, allow token-level temporal modeling to contribute

### Stage 3: Existing Phase 2

Proceed into the current phase-2 regime:

- fusion path remains trainable
- LoRA and any explicitly scheduled modules follow current phase-2 logic

## Loss Design

Keep canonical fusion supervision on `fused_main[B,T,3,H,W]`.

Recommended phase-1 losses:

1. fusion reconstruction loss:
   - `L_fuse_gt = loss(fused_main, GT)`

2. passthrough loss:
   - `L_pass = loss(fused_main, blur_rgb)`
   - use only as an early warmup stabilizer
   - decay it during phase 1 instead of keeping it constant for all phase-1 iterations

3. update penalty:
   - small regularizer on effective update magnitude
   - example target: keep `alpha * sigmoid(gate) * delta_rgb` from growing too fast

Recommended principle:

- output safety should come from `alpha` and a small update penalty
- not from nearly zeroing the entire residual path

## Configuration Changes

The redesign should introduce explicit fusion warmup options.

Example:

```json
"train": {
  "fix_iter": 20000,
  "fusion_warmup": {
    "head_only_iters": 2000,
    "passthrough_weight_start": 0.2,
    "passthrough_weight_end": 0.0,
    "update_penalty_weight": 0.01
  }
}
```

Warmup schedule semantics:

- `current_step < head_only_iters`: stage 1 write-back warmup
- `head_only_iters <= current_step < fix_iter`: stage 2 token-fusion warmup
- `current_step >= fix_iter`: existing phase 2

Recommended new operator options:

```json
"fusion": {
  "operator": "mamba",
  "operator_params": {
    "token_dim": 48,
    "token_stride": 4,
    "num_layers": 3,
    "d_state": 32,
    "d_conv": 4,
    "expand": 2,
    "alpha_init": 0.05,
    "gate_bias_init": -2.0,
    "enable_diagnostics": true
  }
}
```

## Integration With Current Wrapper and Caches

The outer early-fusion contract should remain collapsed for `mamba`.

That means:

- `fused_main == backbone_view`
- no reintroduction of `N*S` execution semantics for `mamba`
- no additional shape guessing in the loss path

Existing semantic cache fields should stay:

- `_last_fusion_main`
- `_last_fusion_exec`
- `_last_fusion_aux`
- `_last_fusion_meta`
- `_last_spike_bins`

Extend `_last_fusion_meta` to include the new diagnostics:

- `token_norm`
- `mamba_norm`
- `delta_norm`
- `gate_mean`
- `effective_update_norm`
- `warmup_stage`

## Testing Strategy

### 1. Operator Semantics Tests

Add tests that verify:

- output shape remains `[B,T,3,H,W]`
- initialization is near passthrough in value space
- initialization is not a degenerate zero-gradient path

### 2. Warmup Schedule Tests

Add tests that verify:

- stage 1 trains only write-back components and `alpha`
- stage 2 correctly unfreezes token encoder and Mamba stack
- backbone and unrelated modules remain frozen during phase 1

### 3. Diagnostics Tests

Add tests that verify:

- forward populates stable diagnostics in fusion metadata or explain hook
- diagnostics exist for normal forward, not only special debug modes

### 4. Integration Tests

Add tests that verify:

- early collapsed wrapper behavior remains intact
- phase-1 fusion aux loss still supervises canonical `T` frames only
- phase transition does not require expanded early semantics

## Acceptance Criteria

The redesign should be considered successful only if it satisfies both interface and optimization criteria.

### Interface acceptance

- `mamba` still builds as an early collapsed operator
- external input and output shapes stay unchanged
- existing early wrapper and VRT integration remain valid

### Optimization acceptance

- within the first `2k` iterations, phase-1 `train/G_loss` shows a clear downward trend relative to startup
- `effective_update_norm` becomes non-zero without exploding
- `gate_mean` does not remain pinned near zero for the entire early warmup
- debug outputs show meaningful but conservative RGB correction before phase 2
- transition into phase 2 does not trigger severe loss instability attributable to the fusion operator

## Alternatives Considered

### 1. Keep current frame-local per-pixel Mamba and only tune initialization

Rejected because it preserves the most optimization-hostile part of the current design: too many independent pixel-level sequences for a conservative residual operator.

### 2. Use a two-axis Mamba over both spike-bin time and frame time

Rejected because it overlaps too strongly with VRT’s temporal role and would substantially increase training complexity.

### 3. Abandon Mamba and return to gated fusion only

Rejected because the goal is to preserve explicit temporal modeling of spike micro-structure inside fusion.

## Risks and Mitigations

### Risk: tokenization loses too much spatial detail

Mitigation:

- keep write-back head spatially aware
- choose conservative token stride
- validate on debug outputs and ablations

### Risk: warmup schedule adds training complexity

Mitigation:

- keep schedule simple and deterministic
- encode warmup stage explicitly in config and metadata
- add schedule unit tests

### Risk: diagnostics become noisy or inconsistent

Mitigation:

- define a stable minimal diagnostic set
- keep diagnostics numeric and cheap to compute
- validate schema in tests

## Implementation Scope

Primary files expected to change in implementation:

- `models/fusion/operators/mamba.py`
- `models/model_plain.py`
- `tests/models/test_fusion_early_adapter.py`
- `tests/models/test_model_plain_fusion_aux_loss.py`
- new or updated tests for warmup scheduling and diagnostics

This scope is still a single implementation unit and does not require decomposition into multiple independent specs.
