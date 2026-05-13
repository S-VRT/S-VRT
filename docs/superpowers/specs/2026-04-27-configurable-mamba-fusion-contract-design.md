# Configurable Mamba Fusion Contract Design

**Date:** 2026-04-27

## Summary

Add a config-level early-fusion contract switch so `mamba` can be ablated as either:

- `collapsed`: fusion models spike-bin micro-time and returns `[B, N, 3, H, W]`
- `expanded`: fusion is applied per spike bin and VRT executes on `[B, N*S, 3, H, W]`

The default remains operator-defined behavior. For current `mamba`, that means `collapsed`.

## Problem

The current fusion code uses two different early-fusion contracts:

- `gated`, `concat`, and `pase` use an expanded contract. The wrapper repeats RGB to match spike bins and sends `N*S` frames to the VRT backbone.
- `mamba` uses a collapsed contract. It consumes structured `rgb[B,N,3,H,W]` and `spike[B,N,S,H,W]`, models spike-bin structure inside fusion, and returns `N` fused RGB frames.

This difference is intentional, but it makes ablation experiments too dependent on implicit operator behavior. We need a config switch that can answer whether high temporal resolution should be modeled inside fusion or by the VRT backbone.

## Goals

1. Keep current `mamba` collapsed behavior as the default.
2. Allow config-only selection of early-fusion frame contract.
3. Support `mamba` expanded ablation without renaming the operator.
4. Preserve existing expanded operators by default.
5. Make metadata report both requested and effective contracts.

## Non-Goals

1. No dataset contract changes.
2. No redesign of middle or hybrid fusion.
3. No change to VRT temporal windows or optical-flow modules beyond existing wrapper alignment.
4. No attempt to make expanded mamba the recommended default.

## Configuration

Add optional field:

```json
"fusion": {
  "early": {
    "frame_contract": "operator_default"
  }
}
```

Allowed values:

- `"operator_default"`: use `operator.frame_contract`
- `"collapsed"`: pass structured `[B,N] + [B,N,S]` tensors to operators that support structured early fusion
- `"expanded"`: repeat RGB per spike bin and run the operator on `N*S`

For backward compatibility, missing `frame_contract` behaves as `"operator_default"`.

## Behavior

### Default Mamba

```json
"operator": "mamba",
"early": {
  "frame_contract": "operator_default"
}
```

Effective behavior:

- input to mamba: `rgb[B,N,3,H,W]`, `spike[B,N,S,H,W]`
- output to VRT: `backbone_view[B,N,3,H,W]`
- fusion metadata: `requested_frame_contract="operator_default"`, `frame_contract="collapsed"`

### Expanded Mamba Ablation

```json
"operator": "mamba",
"early": {
  "frame_contract": "expanded"
}
```

Effective behavior:

- wrapper repeats RGB to `B,N*S,3,H,W`
- wrapper reshapes spike to `B,N*S,1,H,W`
- mamba sees each spike bin as one timestep with `S=1`
- VRT executes on `B,N*S,3,H,W`
- `fused_main` is center-subframe reduced to `B,N,3,H,W`

This is an ablation path, not the preferred semantic path. It tests whether letting VRT model high temporal resolution improves restoration.

### Collapsed Override

```json
"operator": "mamba",
"early": {
  "frame_contract": "collapsed"
}
```

Equivalent to current mamba default, but explicit in config.

## Validation

Rules:

- `frame_contract` must be one of `operator_default`, `collapsed`, or `expanded`.
- `collapsed` requires the operator to support structured early fusion. This is currently true for `mamba`.
- `expanded` is allowed for any early operator that accepts `[B,T,C,H,W]` RGB and spike tensors.
- The existing hard rejection of `mamba + early.expand_to_full_t=true` must only apply when the effective contract is collapsed. In expanded mode, the new `frame_contract` explicitly requests the old full-time execution semantics.

## Testing

Add tests for:

1. Default `mamba` remains collapsed.
2. `frame_contract="expanded"` overrides mamba to expanded packaging.
3. Invalid `frame_contract` raises a clear error.
4. VRT accepts `mamba + early.frame_contract="expanded"` instead of rejecting it because of the old `expand_to_full_t` guard.
5. Fusion metadata includes requested and effective contracts.
