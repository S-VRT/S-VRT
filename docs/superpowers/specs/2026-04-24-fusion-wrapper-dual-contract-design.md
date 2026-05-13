# Fusion Wrapper Dual-Contract Design

**Date:** 2026-04-24

## Summary

Refactor early fusion so the wrapper, not the training loop, owns frame packaging and time-axis semantics.

The early-fusion wrapper should expose three distinct outputs:

- `fused_main`: canonical supervision view with shape `[B, N, C, H, W]`
- `backbone_view`: execution view consumed by downstream flow/VRT backbone, with shape `[B, T_exec, C, H, W]`
- `aux_view`: optional analysis/debug view
- `meta`: lightweight metadata describing how the wrapper derived those views

This design keeps `mamba` as a temporal fusion operator, preserves existing `N*S`-based early-fusion operators, and removes shape guessing from loss code.

## Problem Statement

Current early fusion mixes three different concerns:

1. How operator inputs are packaged
2. Which tensor is supervised against GT
3. Which tensor drives downstream execution

That coupling caused the current `mamba` instability:

- `mamba` naturally outputs a collapsed `[B, N, 3, H, W]` tensor
- phase-1 aux loss historically assumed an expanded `[B, N*S, 3, H, W]` tensor
- VRT forward and loss code inferred tensor meaning from shape and slicing conventions
- when those assumptions diverged, supervision semantics became incorrect

There is also a second architectural constraint:

- existing expanded early-fusion operators do not merely expose `N*S` for debug
- they use the expanded time axis as part of the actual downstream execution path

Therefore the correct solution is not “everything becomes `N`”. The correct solution is to separate supervision time axis from execution time axis.

## Goals

1. Preserve `mamba` as a temporal fusion operator.
2. Make fusion supervision explicitly operate on canonical `N` frames.
3. Preserve existing expanded early-fusion operators whose execution path relies on `N*S`.
4. Move frame packaging, reduction, and contract declaration into the wrapper.
5. Keep existing config files valid by default.

## Non-Goals

1. No redesign of middle or hybrid fusion in this change.
2. No mandatory rewrite of existing expanded operators like `gated`, `concat`, or `pase`.
3. No dataset-side change to the current input contract `rgb[B,N,3,H,W]` + `spike[B,N,S,H,W]`.
4. No required config migration for existing experiment JSON files.

## Core Design

### 1. Separate Supervision View from Execution View

The wrapper must return semantically distinct tensors:

```python
{
    "fused_main": Tensor,      # [B, N, C, H, W]
    "backbone_view": Tensor,   # [B, T_exec, C, H, W]
    "aux_view": Tensor | None,
    "meta": {
        "operator_name": str,
        "frame_contract": str,        # "expanded" | "collapsed"
        "spike_bins": int,
        "main_steps": int,            # N
        "exec_steps": int,            # T_exec
        "aux_steps": int | None,
        "main_from_exec_rule": str | None,
    },
}
```

Rules:

- `fused_main` is the only tensor used for fusion supervision, metrics, and default debug view.
- `backbone_view` is the only tensor used for downstream flow estimation and VRT backbone execution.
- `aux_view` is optional and intended for analysis or visualization only.
- downstream code must never infer semantics from tensor shape alone when `meta` already declares them.

### 2. Two Early-Fusion Contracts

#### Collapsed Contract

Used by operators whose natural execution view already matches the canonical RGB frame axis.

Initial target:

- `mamba`

Behavior:

- operator receives `rgb[B,N,3,H,W]` and `spike[B,N,S,H,W]`
- operator internally models subframe dynamics
- wrapper returns:
  - `fused_main[B,N,C,H,W]`
  - `backbone_view[B,N,C,H,W]`
  - `aux_view=None` unless operator later chooses to expose diagnostics
  - `meta.frame_contract = "collapsed"`

#### Expanded Contract

Used by operators whose downstream execution path naturally works on expanded time axes.

Initial targets:

- `gated`
- `concat`
- `pase`

Behavior:

- wrapper prepares operator-facing expanded inputs
- operator computes its natural expanded output on `N*S`
- wrapper returns:
  - `backbone_view[B,N*S,C,H,W]`
  - `fused_main[B,N,C,H,W]`, derived explicitly from `backbone_view`
  - `aux_view`, optionally equal to `backbone_view` when useful for debug
  - `meta.frame_contract = "expanded"`

This preserves the current operational meaning of expanded early fusion while giving training a stable canonical supervision tensor.

## Wrapper Responsibilities

The early wrapper becomes the single contract layer for:

1. Input packaging
2. Execution-view normalization
3. Canonical supervision-view derivation
4. Metadata declaration

### Input Packaging

Wrapper input remains unchanged:

- `rgb[B,N,3,H,W]`
- `spike[B,N,S,H,W]`

Packaging depends on `frame_contract`:

- `collapsed`: pass original `N` timeline and per-frame `S` subframes directly
- `expanded`: construct expanded operator-facing views on `N*S`

### Execution View

Wrapper must expose the tensor that downstream flow/VRT backbone should actually execute on:

- `collapsed`: `backbone_view = operator_output`
- `expanded`: `backbone_view = expanded_operator_output`

This makes execution semantics explicit instead of piggybacking on whichever tensor happened to come out of fusion.

### Canonical Supervision View

Wrapper must also expose a canonical `fused_main[B,N,C,H,W]`.

Derivation rules:

- `collapsed`: identity (`fused_main = backbone_view`)
- `expanded`: reduce `backbone_view[N*S]` to canonical `N`

The reduction policy must be explicit and owned by the wrapper. It must not live in the training loss code.

Initial compatibility reduction:

- preserve the current center-subframe-per-RGB-frame rule
- encode that in `meta.main_from_exec_rule = "center_subframe"`

### Metadata

Wrapper emits metadata so downstream code can reason explicitly:

- `operator_name`
- `frame_contract`
- `spike_bins`
- `main_steps`
- `exec_steps`
- `aux_steps`
- `main_from_exec_rule`

## Training Path Changes

### Main Principle

Supervision and execution intentionally use different views.

Training rules:

- phase-1 fusion aux loss reads only `fused_main`
- phase-2 fusion aux loss reads only `fused_main`
- downstream flow and VRT backbone execute only on `backbone_view`

### Phase-1 Fusion Aux Loss

Phase-1 supervision becomes:

- compare `fused_main[B,N,...]` directly against `GT[B,N,...]`
- compare `fused_main[B,N,...]` directly against `blur_rgb[B,N,...]` for passthrough regularization

This removes:

- shape guessing in loss code
- implicit `S // 2 :: S` slicing in `ModelPlain`
- silent broadcasting caused by incompatible time dimensions

### Cached Fusion Outputs

Replace ambiguous cache fields like `_last_fusion_out` with explicit fields:

- `_last_fusion_main`
- `_last_fusion_exec`
- `_last_fusion_aux`
- `_last_fusion_meta`
- `_last_spike_bins`

Any training or debug code reading fusion state must use these fields by semantic purpose.

## VRT Forward Integration

`VRT.forward()` should treat wrapper output as a structured object.

For early fusion:

1. call wrapper
2. cache explicit main/exec/aux/meta outputs
3. pass only `backbone_view` into flow estimation and downstream backbone logic

This is the key compatibility rule:

- `fused_main` is not the downstream execution tensor for expanded operators
- `backbone_view` preserves the original `N*S` execution semantics where needed

## Debug and Visualization

Debug should default to the canonical supervision view:

- default debug target: `fused_main`
- when requested, expanded operators may also dump `backbone_view` or `aux_view`
- dump naming/indexing must rely on `meta.frame_contract` and `meta.main_from_exec_rule`, not pure shape inference

This keeps debug informative without contaminating training semantics.

## Compatibility Strategy

Compatibility requirement: existing `N*S`-based early-fusion schemes must keep their downstream behavior.

Chosen strategy:

- `expanded` and `collapsed` contracts coexist explicitly
- wrapper normalizes both into the same structured result
- `expanded` operators retain expanded execution semantics through `backbone_view`
- `collapsed` operators like `mamba` operate naturally on `N`

This avoids both bad extremes:

1. forcing old operators to collapse to `N` and changing their behavior
2. forcing collapsed operators to emit fake `N*S` tensors just to satisfy historical loss code

## Configuration Strategy

This refactor should use config-compatible defaults.

### Existing Configs Must Continue to Work

These existing fields remain valid and unchanged:

- `fusion.operator`
- `fusion.placement`
- `fusion.mode`
- `fusion.out_chans`
- `fusion.operator_params`
- existing debug switches

The wrapper should infer behavior from operator defaults:

- `mamba` defaults to `frame_contract = "collapsed"`
- `gated`, `concat`, and `pase` default to `frame_contract = "expanded"`

### New Config Must Be Optional

New config should only be added for special override cases, not required for existing runs.

Optional future fields may include:

- `fusion.wrapper.main_from_exec`
- `fusion.debug.view`

But default behavior should be fully derivable from operator type and wrapper rules, so existing experiment JSON files remain usable without migration.

## Operator Expectations

### Mamba

`mamba` remains a collapsed-contract early operator.

Desired semantics:

- operate over `S` subframes for each RGB frame
- output canonical execution view on `N`
- optionally expose later diagnostics through `aux_view`

`mamba` must never be forced to masquerade as an expanded `N*S` operator.

### Existing Expanded Operators

`gated`, `concat`, and `pase` remain expanded-contract early operators unless deliberately migrated later.

Desired semantics:

- keep current operator logic as much as possible
- preserve expanded downstream execution through `backbone_view`
- derive canonical `fused_main` in wrapper for supervision
- expose expanded view for debug when needed

## Testing Strategy

### Wrapper Unit Tests

Add wrapper tests that verify:

1. collapsed operator returns:
   - `fused_main[N]`
   - `backbone_view[N]`
   - `aux_view` optional
   - `meta.frame_contract == "collapsed"`
2. expanded operator returns:
   - `fused_main[N]`
   - `backbone_view[N*S]`
   - `aux_view` optional or equal to expanded execution view
   - `meta.frame_contract == "expanded"`

### ModelPlain / Loss Tests

Add or update tests so phase-1 aux loss:

- consumes only `_last_fusion_main`
- rejects time mismatch in `_last_fusion_main`
- never depends on expanded slicing rules

### VRT Integration Tests

Add or update tests to verify:

1. early expanded operator:
   - cached main output is `N`
   - cached exec output is `N*S`
   - downstream execution follows exec steps
2. early collapsed operator:
   - cached main output is `N`
   - cached exec output is `N`
3. downstream flow alignment follows `backbone_view` timeline, not `fused_main`

### Debug Tests

Add tests to verify:

- debug defaults to `fused_main`
- expanded views are dumped only when explicitly requested

## Risks

### Risk 1: Hidden coupling to legacy cache names

Some code may still read `_last_fusion_out`.

Mitigation:

- replace all readers in this refactor
- add tests for new cache-field expectations

### Risk 2: Wrapper reduction ambiguity

Expanded outputs still require deterministic `N*S -> N` derivation.

Mitigation:

- make the rule explicit in wrapper implementation and metadata
- keep center-subframe reduction initially for compatibility

### Risk 3: Config drift

The refactor could accidentally make new wrapper config mandatory.

Mitigation:

- derive defaults from operator contract
- keep new config override-only and optional

## Migration Plan

1. Introduce structured wrapper output with `fused_main`, `backbone_view`, `aux_view`, and `meta`.
2. Encode explicit `expanded` / `collapsed` operator contracts.
3. Update VRT to cache explicit main/exec/aux/meta outputs and execute on `backbone_view`.
4. Update phase-1 and phase-2 fusion supervision to read only `fused_main`.
5. Update debug tooling to default to `fused_main` and optionally expose expanded execution views.
6. Preserve config compatibility by deriving default wrapper behavior from operator type.
7. Lock the new semantics with focused wrapper, loss, and integration tests.

## Expected Outcome

After this refactor:

- `mamba` is trained under the correct `N`-frame supervision contract
- existing `N*S`-based expanded operators keep their downstream execution semantics
- wrapper owns frame packaging and time-axis normalization
- training code becomes semantic and contract-driven instead of shape-guessing
- config files remain compatible by default
