# PASE-Residual Fusion Design

**Date:** 2026-04-28

## Summary

Add a new early-fusion operator named `PASE-Residual Fusion` as a **strong non-sequential baseline** for the spike-fusion study.

The design uses:

- a frame-local RGB encoder
- a true multi-bin `PASE` spike encoder
- a residual write-back head with `delta`, `gate`, and `alpha`

Unlike tokenized `mamba` and `attention`, this operator does **not** include an explicit temporal sequence mixer. Its role is to answer a different question:

- is a strong adaptive spike encoder already sufficient?
- or does explicit sequence modeling still help beyond adaptive non-sequential fusion?

This operator should be `collapsed-only`.

## Problem

The current `gated` operator is too small and too weakly structured to serve as the main non-sequential baseline against `mamba` or `attention`.

The current `pase` fusion operator is also not an appropriate answer, but for a different reason:

- the original `PixelAdaptiveSpikeEncoder` is designed to consume a full spike clip shaped like `[B, L, H, W]`
- in the present early-fusion pipeline, the existing `pase` operator is built with `spike_chans=1`
- under the expanded early-fusion contract, the adapter reshapes `spike[B, T, S, H, W]` into `spike[B, T*S, 1, H, W]`

That means the current `pase` fusion path does **not** use PASE as a multi-bin spike encoder. It instead applies a single-channel PASE-like path independently to each subframe.

This weakens the intended comparison. We do not want a baseline that merely says:

- “what if we process each subframe independently?”

We want a baseline that says:

- “what if we use a strong adaptive spike encoder on the whole per-frame spike clip, but do not use an explicit temporal sequence mixer?”

## Goals

1. Create a strong non-sequential fusion baseline.
2. Let PASE consume the full spike-bin dimension `S` per RGB frame.
3. Keep the output interface aligned with collapsed early fusion:
   - input: `rgb[B, T, 3, H, W]`, `spike[B, T, S, H, W]`
   - output: `fused[B, T, 3, H, W]`
4. Reuse the same residual write-back philosophy as tokenized `mamba`:
   - small but non-zero startup updates
   - learnable `alpha`
   - explicit `delta` and `gate`
5. Make the baseline strong enough that beating it supports the claim that explicit sequence modeling is useful beyond adaptive spike encoding.

## Non-Goals

1. Do not make this operator a fairness-perfect token-mixer match for `mamba` or `attention`.
2. Do not support `expanded` contract in this design.
3. Do not redesign the original `PixelAdaptiveSpikeEncoder`.
4. Do not replace the lightweight `gated` baseline; that operator can still exist as a separate lightweight comparison.

## Naming

Use:

- code/config name: `pase_residual`
- paper name: `PASE-Residual Fusion`

Rationale:

- `PASE` already carries adaptive/selective meaning
- the key architectural add-on is a residual write-back head
- avoiding `gated` in the name prevents confusion with the existing small `gated` operator

## Design Decision

Implement `PASE-Residual Fusion` as a collapsed structured early-fusion operator that explicitly consumes the full per-frame spike clip before writing a conservative RGB residual.

This keeps the comparison clean:

- `mamba` / `attention`: explicit sequence modeling inside fusion
- `PASE-Residual Fusion`: strong adaptive spike encoding without explicit sequence modeling

The baseline is intentionally **not** a contract study. Its job is not to test whether high temporal resolution should be handed to VRT. Its job is to test whether an adaptive non-sequential spike encoder is already enough.

## Architecture

### 1. RGB Context Encoder

Use a lightweight RGB encoder that maps each RGB frame to a feature map:

- input: `rgb_flat[B*T, 3, H, W]`
- output: `rgb_ctx[B*T, C, H, W]` or optionally low-resolution `rgb_ctx[B*T, C, H', W']`

Recommended default:

- `Conv2d(3 -> C, 3x3, padding=1)`
- `ReLU`
- `Conv2d(C -> C, 3x3, padding=1)`
- `ReLU`

This encoder does not need to be identical to tokenized `mamba`, but it should be strong enough that the baseline is not trivially capacity-limited.

### 2. PASE Spike Encoder

This is the core of the design.

For each RGB frame:

- reshape `spike[B, T, S, H, W]` to `spike_flat[B*T, S, H, W]`
- run `PixelAdaptiveSpikeEncoder(in_chans=S, out_chans=C, ...)`

Output:

- `pase_feat[B*T, C, H, W]`

Meaning:

- PASE now consumes the full spike clip per frame, which matches its original intended semantics
- the baseline gains strong adaptive spike encoding without introducing an explicit sequential mixer block

### 3. Fusion Body

Fuse RGB and PASE features with a lightweight convolutional body:

- concatenate `rgb_ctx` and `pase_feat`
- run a small fusion body, such as:
  - `Conv2d(2C -> C, 3x3, padding=1)`
  - `ReLU`
  - optional second `Conv2d(C -> C, 3x3, padding=1)`
  - `ReLU`

This body should remain moderate in size. We want a strong baseline, not a completely unconstrained fusion subnet.

### 4. Residual Write-Back Head

Use the same residual write-back idea as tokenized `mamba`:

- `delta = Conv2d(C -> 3, 1x1)`
- `gate_logits = Conv2d(C -> 3, 1x1)`
- `gate = sigmoid(gate_logits)`
- `effective_update = alpha * gate * delta`
- `out = rgb + effective_update`

Initialization should follow the current stabilized fusion design:

- small random init for `delta`
- small random init for `gate`
- mild negative `gate_bias_init`, e.g. `-2.0`
- explicit `alpha_init`, e.g. `0.05`

This keeps startup behavior conservative without collapsing gradients.

## Why Collapsed-Only

`PASE-Residual Fusion` should be a structured early operator with:

- `expects_structured_early = True`
- `frame_contract = "collapsed"`

Reason:

PASE is meaningful here precisely because it consumes the full spike-bin dimension `S` per RGB frame.

If this operator were routed through the existing expanded path:

- RGB would be repeated from `T` to `T*S`
- spike would be reshaped from `[B, T, S, H, W]` to `[B, T*S, 1, H, W]`

Then PASE would no longer see the full spike clip. It would instead process one subframe at a time with a single input channel, which defeats the purpose of using PASE as a strong adaptive spike encoder.

So this design intentionally rejects expanded semantics.

## Forward Semantics

Input:

- `rgb[B, T, 3, H, W]`
- `spike[B, T, S, H, W]`

Processing:

1. flatten RGB frames to `rgb_flat[B*T, 3, H, W]`
2. flatten spike clips to `spike_flat[B*T, S, H, W]`
3. compute `rgb_ctx = rgb_context_encoder(rgb_flat)`
4. compute `pase_feat = pase(spike_flat)`
5. concatenate RGB and spike features
6. run fusion body
7. compute `delta`, `gate`, and `effective_update`
8. reshape back to `[B, T, 3, H, W]`
9. return `rgb + effective_update`

External meaning:

- one fused RGB frame per original RGB frame
- no `N*S` execution path inside this operator
- no explicit spike-bin temporal mixer module

## Training and Warmup

To keep training behavior comparable with tokenized `mamba`, this operator should support:

- `set_warmup_stage(stage)`
- `diagnostics()`

Recommended warmup behavior:

- `writeback_only`
  - freeze `rgb_context_encoder`
  - freeze `pase`
  - freeze fusion body
  - train only write-back head and `alpha`
- `token_mixer`
  - interpret this stage generically as `full_feature_fusion`
  - unfreeze RGB encoder, PASE encoder, and fusion body
  - keep write-back trainable
- `full`
  - all trainable

Even though this operator has no temporal mixer, reusing the same stage names reduces training-logic branching in `ModelPlain`.

## Diagnostics

Expose a compact diagnostic set analogous to `mamba`:

- `pase_norm`
- `fusion_body_norm`
- `delta_norm`
- `gate_mean`
- `effective_update_norm`
- `warmup_stage`

These should flow through the existing fusion metadata path into `ModelPlain.current_log()`.

## Parameter and Capacity Guidance

This baseline does not need to be architecturally identical to `mamba` or `attention`, but it should be in the same rough capacity regime.

Recommendations:

1. choose fusion width `C` so total parameters are not dramatically smaller than the temporal baselines
2. if needed, increase fusion body width before making the architecture deeper
3. report actual `params`, `FLOPs`, iteration time, and memory

The goal is:

- strong enough that winning against it means something
- still simple enough that its lack of explicit sequence modeling remains the central difference

## Experimental Role

`PASE-Residual Fusion` should appear in the paper as:

- a **strong non-sequential baseline**

It should not be described as:

- the fairness-perfect counterpart to `mamba`
- a contract ablation
- an expanded-time fusion baseline

The question it answers is:

- does a strong adaptive spike encoder plus residual fusion already solve the problem well enough?

If `mamba` or `attention` still outperform it, that supports the argument that explicit spike-sequence modeling adds value beyond adaptive spike encoding alone.

## Testing Strategy

### Operator Tests

Add tests that verify:

1. the operator accepts structured early inputs `rgb[B,T,3,H,W]` and `spike[B,T,S,H,W]`
2. output shape remains `[B, T, 3, H, W]`
3. initialization gives small but non-zero updates
4. gradients reach write-back parameters at startup
5. `set_warmup_stage("writeback_only")` freezes PASE and feature encoders
6. `set_warmup_stage("token_mixer")` unfreezes them again

### Contract Tests

Add tests that verify:

1. default contract is collapsed
2. trying to force expanded mode raises a clear error or is rejected during build
3. metadata reports `frame_contract="collapsed"`

### Integration Tests

Add tests that verify:

1. VRT builds with `fusion.operator="pase_residual"`
2. phase-1 fusion-only forward caches `fusion_main`, `fusion_exec`, and metadata correctly
3. diagnostics reach training logs

## Acceptance Criteria

The design is successful if:

1. PASE truly consumes the full spike-bin dimension `S`
2. the operator acts as a collapsed early-fusion module
3. it is materially stronger than the current tiny `gated` baseline
4. its main remaining difference from temporal baselines is the absence of an explicit sequence mixer
5. its role in the paper is unambiguous: strong non-sequential baseline

## Risks and Mitigations

### Risk: the baseline becomes too strong by adding too much generic fusion capacity

Mitigation:

- keep the fusion body shallow
- avoid turning the operator into a deep generic CNN

### Risk: the baseline is still too weak because PASE is the only strong component

Mitigation:

- tune feature width `C`
- use the stabilized write-back head instead of the old tiny gated residual head

### Risk: warmup stage names become semantically awkward for a non-sequential operator

Mitigation:

- reuse stage names for training simplicity
- document that `token_mixer` means “full feature fusion stage” for this operator

## Implementation Scope

Primary files expected to change:

- `models/fusion/operators/pase_residual.py` (new)
- `models/fusion/operators/__init__.py`
- `models/fusion/factory.py` if needed for registration
- `tests/models/test_fusion_early_adapter.py`
- `tests/models/test_vrt_fusion_integration.py`
- `tests/models/test_two_phase_training.py`
- ablation config files under `options/`

This remains a single implementation unit and is ready to be turned into an implementation plan after review.
