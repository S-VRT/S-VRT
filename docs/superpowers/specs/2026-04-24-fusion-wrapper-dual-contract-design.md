# Fusion Wrapper Dual-Contract Design

**Date:** 2026-04-24

## Summary

Refactor early fusion so the wrapper, not the training loop, owns frame packaging and output-shape semantics.

The new early-fusion contract is a dual-output interface:

- `fused_main`: canonical fused frames with shape `[B, N, C, H, W]`
- `aux_view`: optional operator-specific auxiliary view, commonly `[B, N*S, C, H, W]`
- `meta`: lightweight metadata describing the contract and frame semantics

This design keeps `mamba` as a temporal fusion operator while preserving existing early-fusion schemes that still depend on expanded `N*S` behavior.

## Problem Statement

Current early fusion mixes three responsibilities across multiple layers:

1. Operator-specific frame packaging
2. Main training supervision semantics
3. Debug and visualization semantics

That coupling caused the current `mamba` instability:

- `mamba` produces a collapsed `[B, N, 3, H, W]` output
- phase-1 fusion aux loss historically assumed expanded `[B, N*S, 3, H, W]`
- training code inferred semantics from tensor shape and slicing rules
- when the assumption was wrong, the loss path silently degraded into incorrect supervision

The root architectural issue is not only the `mamba` implementation. The actual problem is that the training path has no explicit contract for what fusion returns.

## Goals

1. Preserve `mamba` as a temporal fusion operator.
2. Make main training supervision shape-invariant and explicit.
3. Move frame packaging and reduction into the fusion wrapper.
4. Preserve existing early-fusion operators that rely on expanded `N*S` semantics.
5. Keep debug and visualization able to inspect expanded or operator-specific intermediate views.

## Non-Goals

1. No redesign of middle fusion or hybrid fusion semantics in this change.
2. No mandatory rewrite of existing expanded operators like `gated`, `concat`, or `pase`.
3. No change to dataset-side spike packaging contract (`rgb[N]`, `spike[N,S]` remains the input).
4. No redesign of phase-2 LoRA or full VRT restoration behavior beyond consuming the new wrapper outputs.

## Design Overview

### 1. Canonical Wrapper Output

Early fusion wrapper returns a structured result instead of a bare tensor.

Canonical fields:

```python
{
    "fused_main": Tensor,   # [B, N, C, H, W]
    "aux_view": Tensor | None,
    "meta": {
        "operator_name": str,
        "frame_contract": str,   # "expanded" | "collapsed"
        "spike_bins": int,
        "main_steps": int,
        "aux_steps": int | None,
    },
}
```

Rules:

- `fused_main` is always the only tensor used by main training and backbone ingress.
- `aux_view` is optional and never required for correctness of training.
- `meta` carries explicit semantics so downstream code never infers meaning from shape alone.

### 2. Two Early-Fusion Contracts

#### Collapsed Contract

Used by operators whose natural output is already `[B, N, C, H, W]`.

Initial target:

- `mamba`

Behavior:

- operator receives `rgb[B,N,3,H,W]` and `spike[B,N,S,H,W]`
- operator performs per-frame sub-sequence temporal modeling internally
- wrapper returns:
  - `fused_main = operator_output`
  - `aux_view = optional operator-specific diagnostic tensor or None`
  - `meta.frame_contract = "collapsed"`

#### Expanded Contract

Used by operators that naturally work on subframe-expanded time axes.

Initial targets:

- `gated`
- `concat`
- `pase`

Behavior:

- wrapper prepares the operator input in its expected expanded form
- operator may continue to compute over `N*S`
- wrapper reduces the expanded result into canonical `fused_main[B,N,C,H,W]`
- wrapper preserves the original expanded tensor in `aux_view`
- `meta.frame_contract = "expanded"`

This preserves existing `N*S`-dependent schemes without forcing them to adopt collapsed semantics.

## Wrapper Responsibilities

The early-fusion wrapper becomes the single contract layer for:

1. Input packaging
2. Output reduction
3. Metadata declaration

### Input Packaging

Wrapper input remains:

- `rgb[B,N,3,H,W]`
- `spike[B,N,S,H,W]`

Packaging behavior depends on `frame_contract`:

- `collapsed`: pass original `N` timeline and `S` subframes directly
- `expanded`: construct the operator-facing expanded timeline or equivalent flattened view

Main training code does not know or care which path was used.

### Output Reduction

Wrapper must guarantee canonical `fused_main[B,N,C,H,W]`.

Reduction rules:

- `collapsed`: identity reduction
- `expanded`: explicit wrapper-owned reduction from `N*S` to `N`

The reduction strategy must be explicit and deterministic. It cannot be encoded as hidden slicing logic in training loss code.

Initial reduction rule for compatibility:

- preserve the current “center subframe per RGB frame” behavior for expanded early-fusion outputs

That rule moves out of `ModelPlain` and into wrapper-owned reduction logic.

### Metadata

Wrapper emits `meta` so downstream code can reason explicitly:

- `operator_name`
- `frame_contract`
- `spike_bins`
- `main_steps`
- `aux_steps`
- optional fields later if needed, such as `reduction_mode`

## Training Path Changes

### Main Principle

Training must consume only `fused_main`.

This applies to:

- phase-1 fusion-only fast path
- full VRT backbone ingress
- flow alignment time-axis selection
- phase-1 and phase-2 fusion auxiliary losses

### Phase-1 Fusion Aux Loss

Phase-1 supervision becomes:

- compare `fused_main[B,N,...]` directly against `GT[B,N,...]`
- compare `fused_main[B,N,...]` directly against `blur_rgb[B,N,...]` for passthrough regularization

This removes:

- shape guessing
- `S // 2 :: S` slicing in the loss path
- accidental broadcasting caused by mismatched time dimensions

### Cached Fusion Outputs

Replace ambiguous cached state:

- `_last_fusion_out`

With explicit cached state:

- `_last_fusion_main`
- `_last_fusion_aux`
- `_last_fusion_meta`
- `_last_spike_bins`

Any training or debug code reading fusion intermediates must read one of these explicit fields.

## VRT Forward Integration

`VRT.forward()` should treat wrapper output as a structured object.

For early fusion:

1. call wrapper
2. store explicit cached outputs
3. feed only `fused_main` into downstream flow and backbone logic

Flow alignment behavior:

- flow path should align with `fused_main.size(1)` only
- expanded `aux_view` is never used as the operational time axis for backbone execution

This ensures backbone and restoration path always operate on the canonical `N` timeline.

## Debug and Visualization

Debug remains able to inspect fine-grained views without polluting training semantics.

Rules:

- default debug target is `fused_main`
- when `aux_view` exists and debug config requests subframe inspection, dump `aux_view`
- naming and indexing rules should use `meta.frame_contract` instead of inferring from tensor shape

This preserves visibility into expanded early fusion while keeping training semantics stable.

## Compatibility Strategy

Compatibility requirement: do not break existing `N*S`-dependent early-fusion schemes.

Chosen strategy:

- explicit split between `expanded` and `collapsed` early contracts
- wrapper normalizes both into the same dual-output result
- old expanded operators keep their natural behavior
- new collapsed operators like `mamba` do not need to fake `N*S`

This avoids the two common failure modes:

1. forcing old operators into an unnatural `N` output
2. forcing collapsed operators to emit synthetic `N*S` tensors just to satisfy historical training code

## Operator Expectations

### Mamba

`mamba` remains a collapsed-contract early operator.

Desired semantics:

- operate over the `S` subframes for each RGB frame
- return `fused_main[B,N,3,H,W]`
- optionally expose intermediate diagnostics through `aux_view` or metadata later

No training code may assume `mamba` outputs expanded `N*S`.

### Existing Expanded Operators

`gated`, `concat`, and `pase` continue as expanded-contract early operators unless explicitly migrated later.

Desired semantics:

- keep current operator logic as much as possible
- wrapper reduces their expanded output to canonical `fused_main`
- debug can still inspect their full `aux_view`

## Testing Strategy

### Unit Tests

Add contract-focused wrapper tests covering:

1. collapsed operator returns:
   - `fused_main[N]`
   - `aux_view` optional
   - `meta.frame_contract == "collapsed"`
2. expanded operator returns:
   - `fused_main[N]`
   - `aux_view[N*S]`
   - `meta.frame_contract == "expanded"`

### ModelPlain / Loss Tests

Add or update tests so phase-1 aux loss:

- consumes only `_last_fusion_main`
- ignores `aux_view` for supervision
- fails loudly on incompatible shapes instead of broadcasting

### VRT Integration Tests

Add or update tests to verify:

1. early expanded operator:
   - cached main output is `N`
   - cached aux output is `N*S`
2. early collapsed operator:
   - cached main output is `N`
   - cached aux output is optional
3. downstream flow and backbone path follow `fused_main` timeline only

### Regression Tests

Preserve existing behavior for current expanded early operators:

- no operator-level regression in expected output path
- no loss regression from historical expanded schemes

## Risks

### Risk 1: Hidden coupling to `_last_fusion_out`

Other code may still read the old field.

Mitigation:

- replace all readers in this refactor
- add focused tests for cached fusion state names and shapes

### Risk 2: Wrapper reduction policy ambiguity

Expanded outputs still need a deterministic `N*S -> N` policy.

Mitigation:

- make reduction explicit in wrapper metadata and implementation
- keep current center-subframe rule initially for compatibility

### Risk 3: Debug tooling may assume tensor-only fusion outputs

Current debug code may expect a bare tensor and infer frame semantics.

Mitigation:

- adapt debug entry points to explicit cached fields
- route subframe-specific dumps through `aux_view`

## Migration Plan

1. Introduce a structured dual-output result for early fusion wrapper.
2. Teach wrapper to normalize `expanded` and `collapsed` operators.
3. Update `VRT.forward()` and phase-1 fast path to cache explicit main/aux/meta outputs.
4. Update fusion aux loss to consume only canonical `fused_main`.
5. Update debug and visualization to use `aux_view` when requested.
6. Update tests to encode the new contract.

## Expected Outcome

After this refactor:

- `mamba` is trained under the correct `N`-frame supervision contract
- existing `N*S`-based early-fusion operators still work
- wrapper owns frame packaging and reduction
- training code becomes contract-driven instead of shape-guessing
- debug retains access to expanded intermediate views without contaminating main training logic
