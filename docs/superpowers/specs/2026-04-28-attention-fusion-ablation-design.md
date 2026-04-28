# Attention Fusion Ablation Design

**Date:** 2026-04-28

## Summary

Add an `attention` early-fusion operator that is intentionally aligned with the current tokenized `mamba` design:

- shared `rgb_context_encoder`
- shared `spike_token_encoder`
- shared `fusion_writeback_head`
- shared warmup and diagnostics interfaces
- different temporal mixer only

The main ablation compares `attention` and `mamba` under the same **collapsed** early-fusion contract. The operator should still be compatible with both `collapsed` and `expanded` frame contracts through the existing `EarlyFusionAdapter`, so later contract ablations do not require an interface redesign.

## Problem

We need a stronger temporal baseline than the current small `gated` operator to defend the claim that `mamba` is beneficial for spike-sequence modeling inside early fusion.

The comparison must avoid two common confounds:

1. changing the temporal mixer and the rest of the operator at the same time
2. changing the temporal mixer and the early-fusion frame contract at the same time

If `attention` uses a different encoder or write-back path, then any gain or loss is hard to attribute to the mixer. If `attention` is evaluated under a different `collapsed/expanded` contract from `mamba`, then any gain or loss is hard to attribute to who performs the temporal aggregation: the fusion operator or the VRT backbone.

## Goals

1. Add `attention` as a first-class early-fusion operator.
2. Keep the operator as structurally close to tokenized `mamba` as possible.
3. Support both `collapsed` and `expanded` frame contracts via the existing early-fusion wrapper path.
4. Make `collapsed attention vs collapsed mamba` the primary ablation.
5. Preserve warmup staging and diagnostics so training behavior can be compared fairly.

## Non-Goals

1. Do not redesign `gated` in this change.
2. Do not decide the final `expanded vs collapsed` winner in this change.
3. Do not change dataset payload structure or VRT backbone semantics.
4. Do not introduce a separate custom wrapper just for attention.

## Design Decision

Implement `AttentionFusionOperator` as a tokenized, RGB-conditioned, frame-local early-fusion operator whose external behavior mirrors `MambaFusionOperator`.

The operator itself owns only the shared tokenization, temporal mixing, and write-back logic. The `collapsed/expanded` contract switch remains the responsibility of `EarlyFusionAdapter`, exactly as it already is for configurable `mamba`.

This keeps the experimental axis clean:

- **mixer ablation:** `attention` vs `mamba` under the same collapsed contract
- **contract ablation:** `collapsed` vs `expanded` for a chosen operator later

## Architecture

### Shared Structure

`AttentionFusionOperator` should follow the same high-level layout as the current tokenized `mamba`:

1. `rgb_context_encoder`
2. `spike_token_encoder`
3. `attention_token_mixer`
4. `fusion_writeback_head`
5. `alpha`
6. `set_warmup_stage()` and `diagnostics()`

The only intended algorithmic difference between `attention` and `mamba` is item 3.

### RGB Context Encoder

Reuse the current lightweight RGB token encoder pattern:

- `Conv2d(3 -> D, stride=token_stride, kernel=3, padding=1)`
- `ReLU`
- `Conv2d(D -> D, kernel=3, padding=1)`
- `ReLU`

Responsibilities:

- supply RGB-conditioned context for fusion
- reduce spatial resolution before token mixing
- keep parameters and compute comparable to `mamba`

### Spike Token Encoder

Reuse the current spike token encoder pattern:

- `Conv2d(1 -> D, stride=token_stride, kernel=3, padding=1)`
- `ReLU`
- `Conv2d(D -> D, kernel=3, padding=1)`

Responsibilities:

- map each spike bin to a low-resolution token grid
- preserve the spike-bin axis so the mixer operates along micro-time

### Attention Token Mixer

Each block should be a lightweight Transformer-style token mixer that runs along the spike-bin axis only.

Recommended block:

1. `LayerNorm`
2. `MultiheadAttention(batch_first=True)`
3. residual add
4. `LayerNorm`
5. `MLP` with `GELU`
6. residual add

Recommended defaults:

- `num_heads = 4` when `token_dim = 48`
- `mlp_ratio = 2.0`
- `num_layers` aligned with `mamba`
- `attn_drop = 0.0`
- `proj_drop = 0.0`

The MLP is important because a raw attention-only block is usually too small relative to the current `mamba` block. Adding a modest feed-forward path makes the budget more comparable without turning this into a full-blown Transformer redesign.

### Write-Back Head

Reuse the same write-back structure and initialization rules as tokenized `mamba`:

- `body`: `Conv2d(D -> D, 3x3) + ReLU`
- `delta`: `Conv2d(D -> 3, 1x1)`
- `gate`: `Conv2d(D -> 3, 1x1)`
- `alpha`: learnable residual scale

Initialization should also match the current `mamba` design:

- small random init for `delta` and `gate`
- mild negative `gate` bias, such as `-2.0`
- small positive `alpha_init`, such as `0.05`

## Forward Semantics

### Collapsed Contract

Input:

- `rgb[B, T, 3, H, W]`
- `spike[B, T, S, H, W]`

Processing:

1. encode RGB to `rgb_low[B, T, D, H', W']`
2. encode spikes to `spike_low[B, T, S, D, H', W']`
3. pack per-location sequences to `seq[B*T*H'*W', S, D]`
4. inject RGB context by adding `rgb_tokens`
5. run `attention_token_mixer` on the `S` axis
6. pool over `S` to get one fused token grid per RGB frame
7. apply write-back head and upsample to `[B, T, 3, H, W]`

Meaning:

- the fusion operator itself performs spike-bin temporal aggregation
- VRT sees already-collapsed frame-aligned fused RGB

### Expanded Contract

The operator should not implement a separate expanded-specific internal algorithm. Instead, it should rely on the existing `EarlyFusionAdapter` contract path, just like configurable `mamba`.

That means the adapter performs:

- RGB repeat from `T` to `T*S`
- spike reshape from `[B, T, S, H, W]` to `[B, T*S, 1, H, W]`

The operator then sees a standard early-fusion input:

- `rgb[B, T*S, 3, H, W]`
- `spike[B, T*S, 1, H, W]`

Consequences:

- inside the operator, the local spike-bin axis length becomes `1`
- the operator still performs tokenization and write-back, but no longer performs multi-bin temporal aggregation inside the mixer
- high temporal resolution is intentionally handed off to the VRT backbone

This is not a bug. It is the meaning of the expanded contract in the current system.

## Interfaces

`AttentionFusionOperator` should expose the same training and diagnostics hooks as `mamba`:

- `expects_structured_early = True`
- `frame_contract = "collapsed"`
- `set_warmup_stage(stage)`
- `diagnostics()`
- optional timer hooks if useful for profiling parity

Recommended diagnostics:

- `token_norm`
- `attention_norm`
- `delta_norm`
- `gate_mean`
- `effective_update_norm`
- `warmup_stage`

`attention_norm` is the direct analog of `mamba_norm`.

## Configuration

Add a new operator choice:

```json
"fusion": {
  "operator": "attention",
  "operator_params": {
    "token_dim": 48,
    "token_stride": 4,
    "num_layers": 3,
    "num_heads": 4,
    "mlp_ratio": 2.0,
    "alpha_init": 0.05,
    "gate_bias_init": -2.0,
    "enable_diagnostics": true
  },
  "early": {
    "frame_contract": "operator_default"
  }
}
```

Recommended defaults:

- `frame_contract = "operator_default"`
- `operator_default` resolves to `collapsed` for `attention`, matching `mamba`

This keeps the main ablation explicit and avoids silently comparing different contracts.

## Parameter and Cost Alignment

The comparison target is **comparable budget**, not exact equality.

Alignment rules:

1. keep `token_dim`, `token_stride`, and `num_layers` shared with `mamba`
2. choose `num_heads` and `mlp_ratio` so per-block parameters are in the same regime as the current `mamba` block
3. report actual `params`, `FLOPs`, iteration time, and peak memory in the ablation table

Practical recommendation:

- start with `token_dim=48`, `num_heads=4`, `mlp_ratio=2.0`, `num_layers=3`
- if parameter count is too low relative to `mamba`, first increase `mlp_ratio`
- do not inflate attention depth independently from `mamba`

## Experimental Plan

### Primary Mixer Ablation

Run first:

1. `mamba + collapsed`
2. `attention + collapsed`

This is the main paper-quality comparison because it isolates the temporal mixer under the same early-fusion contract.

### Secondary Contract Ablation

Only after the primary mixer comparison is understood, run:

1. chosen operator + `collapsed`
2. chosen operator + `expanded`

This answers a different question:

- should spike micro-time be aggregated inside fusion, or handed off to VRT?

It should not be mixed into the main attention-vs-mamba claim.

### Optional Expanded Attention Readiness

Even if expanded attention is not trained immediately, the interface should support it from day one so later ablations do not require API churn or test rewrites.

## Testing Strategy

### Operator Tests

Add tests that verify:

1. `attention` builds with the same constructor shape conventions as `mamba`
2. collapsed forward returns `[B, T, 3, H, W]`
3. diagnostics contain the expected scalar fields
4. initialization produces small but non-zero effective updates and live gradients
5. `set_warmup_stage("writeback_only")` freezes encoder and mixer params while keeping write-back trainable
6. `set_warmup_stage("token_mixer")` unfreezes encoder and mixer params

### Contract Tests

Add tests that verify:

1. `attention` default contract resolves to collapsed
2. `frame_contract="expanded"` routes through the existing adapter behavior
3. expanded `attention` produces `backbone_view` with `T*S` steps and `fused_main` with `T` steps
4. metadata reports requested and effective contracts

### Integration Tests

Add tests that verify:

1. VRT builds and runs with `fusion.operator="attention"`
2. phase-1 fusion-only forward caches `fusion_main`, `fusion_exec`, and `fusion_meta` correctly
3. diagnostics reach `ModelPlain.current_log()` through the existing metadata path

## Acceptance Criteria

The design is successful if all of the following hold:

1. `attention` can replace `mamba` through config only
2. the main collapsed comparison changes only the temporal mixer
3. expanded support does not require a separate operator API
4. diagnostics and warmup staging remain comparable between `attention` and `mamba`
5. later `expanded vs collapsed` experiments can be added without refactoring the wrapper contract

## Risks and Mitigations

### Risk: attention is materially under-parameterized

Mitigation:

- include an MLP in each block
- compare real parameter counts before locking defaults

### Risk: expanded attention is misinterpreted as an attention weakness

Mitigation:

- state clearly that expanded mode changes the role of fusion
- treat expanded results as a contract study, not a pure mixer study

### Risk: later experiments conflate mixer quality and backbone temporal modeling

Mitigation:

- keep the collapsed mixer ablation as the primary result
- present contract ablations separately in paper text and tables

## Implementation Scope

Primary files expected to change:

- `models/fusion/operators/attention.py` (new)
- `models/fusion/operators/__init__.py`
- `models/fusion/factory.py` if needed for registration plumbing
- `tests/models/test_fusion_early_adapter.py`
- `tests/models/test_vrt_fusion_integration.py`
- `tests/models/test_two_phase_training.py`
- `options/...` config files for ablation runs

This remains a single implementation unit and does not require decomposition before planning.
