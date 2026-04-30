# Dual-Scale Temporal Mamba Design

**Date:** 2026-04-30

## Summary

Redesign the current early-fusion `mamba` path so it can exploit raw spike temporal structure instead of only mixing a short TFP bin axis.

The new operator, tentatively named `Dual-Scale Temporal Mamba`, keeps the existing structured early-fusion contract:

- input: `rgb[B,T,3,H,W]`, `spike[B,T,S,H,W]`
- output: `fused[B,T,3,H,W]`

However, unlike the current collapsed Mamba path, it changes both the spike representation and the temporal organization:

- current path: per-frame `TFP bins[S=4,H,W]` -> short-sequence Mamba over `S`
- new path: per-frame `raw_window[L,H,W]` -> local temporal Mamba over `L` -> global temporal Mamba over `T`

This makes the Mamba operator explicitly hierarchical:

1. local Mamba models intra-frame raw spike micro-timing
2. global Mamba models inter-frame temporal evolution
3. a conservative residual write-back updates RGB features

## Problem

The current early-fusion Mamba path does not use Mamba in the regime where its strengths are most relevant.

Today:

- each RGB frame is paired with a spike representation reconstructed as `TFP bins`
- the default configuration uses only `4` spike bins
- the Mamba operator receives `spike[B,T,S,H,W]`
- after token packing, the effective Mamba sequence length is `S`, not `T`

That means the current operator is mostly acting as a per-frame short token mixer over `4` TFP bins, not as a true long-range temporal model.

This creates two issues:

1. the spike input is already temporally compressed before Mamba sees it
2. the operator does not explicitly separate intra-frame micro-temporal structure from inter-frame video dynamics

As a result, current comparisons against `pase_residual` do not fairly test Mamba's temporal modeling ceiling on this task.

## Goals

1. Redesign the Mamba fusion path so it can consume raw per-frame spike windows instead of only TFP bins.
2. Explicitly separate local spike micro-timing from global cross-frame temporal modeling.
3. Keep the operator compatible with the existing early-fusion dual-input contract.
4. Stay within a training budget that can realistically be completed in the current project timeline.
5. Produce a stronger Mamba baseline for comparison against:
   - current collapsed Mamba
   - `pase_residual + tfp`
   - `pase_residual + raw_window`

## Non-Goals

1. Do not redesign the downstream VRT backbone in this phase.
2. Do not flatten the entire `(T,L,H,W)` structure into one giant sequence unless a later ablation justifies it.
3. Do not turn the fusion module into a full replacement restoration backbone.
4. Do not optimize for strict parameter matching with `pase_residual`; optimize for trainable budget realism instead.
5. Do not require the operator to outperform `pase_residual` in the first iteration to count as informative.

## Design Decision

Use a hierarchical two-stage Mamba fusion operator:

- Stage A: `Local Temporal Mamba`
- Stage B: `Global Temporal Mamba`

Rationale:

- it preserves Mamba as the main modeling primitive at both temporal scales
- it aligns with the intuition behind local-to-global temporal decomposition
- it avoids the instability and budget blow-up of a single huge joint `(T,L)` sequence
- it keeps the final fusion head comparable to the existing residual write-back design

This design is conceptually similar to local/global chunked modeling ideas, but remains architecturally grounded in state-space modeling rather than attention.

## Current Behavior

The present collapsed Mamba path can be summarized as:

1. each RGB frame loads a spike clip
2. dataset reconstructs that clip into `TFP bins[S,H,W]`
3. operator encodes each bin independently as a spatial token stream
4. tokens are packed so Mamba runs over the short bin axis `S`
5. the mixed sequence is pooled and written back to RGB

This means:

- Mamba does not see raw spike sequences
- Mamba does not explicitly model cross-frame temporal structure inside the fusion operator
- the effective temporal sequence for Mamba is often only length `4`

## Proposed Representation Model

Use raw centered spike windows as the primary spike representation for the new Mamba path.

Recommended dataset output:

- `rgb[B,T,3,H,W]`
- `spike[B,T,L,H,W]`

Where:

- `L = raw_window_length`
- `L` is a positive odd integer
- the window is centered using the same temporal reference currently used by the TFP path

This keeps time alignment fair across ablations while allowing the operator to observe real spike micro-timing before temporal compression.

## Temporal Decomposition

The new operator should split temporal modeling into two scales.

### 1. Local Temporal Scale

For each RGB frame independently:

- consume `raw_window[L,H,W]`
- model only the intra-frame spike temporal axis `L`
- learn short-range micro-temporal structure such as local firing bursts, motion onset, and asymmetric exposure dynamics

### 2. Global Temporal Scale

After local processing:

- summarize each frame's local spike dynamics into compact tokens
- model only the frame axis `T`
- learn longer-range video evolution across neighboring RGB frames

This decomposition maps naturally onto the task:

- `L` captures sub-frame temporal detail
- `T` captures frame-level temporal consistency and evolution

## Token Organization

Do not flatten `(T,L,H,W)` directly into one monolithic sequence.

Instead:

1. spatially patchify or downsample spike inputs
2. run local temporal modeling over `L`
3. summarize local temporal outputs
4. run global temporal modeling over `T`

Recommended tensor flow:

1. spike input:
   - `spike[B,T,L,H,W]`
2. lightweight spatial projection / patchify:
   - `spike_tok[B,T,L,P,C0]`
   - `P` = number of spatial tokens
   - `C0` = projected token width
3. local temporal Mamba over `L`:
   - `local_feat[B,T,L,P,C1]`
4. temporal summary over `L`:
   - `frame_summary[B,T,P,C1]`
5. global temporal Mamba over `T`:
   - `global_feat[B,T,P,C2]`
6. spatial restore / upsample to feature map:
   - `spike_ctx[B,T,Cf,H,W]`

Why this organization:

- Mamba gets a meaningful raw temporal axis at stage 1
- stage 2 sequence length is only `T`, which is budget-friendly
- `P` acts as a parallel token batch, not as sequence length inside one selective scan

## Operator Architecture

### Local Temporal Mamba

Input:

- `spike_tok[B,T,L,P,C0]`

Execution:

- fix `(b,t,p)`
- run Mamba along the `L` axis

Recommended block:

- pre-norm
- Mamba over `L`
- residual
- lightweight FFN or linear projection
- residual

Guidelines:

- use `1-2` local layers in the first iteration
- local stage should be stronger than the global stage
- avoid excessive depth because raw spike windows can be noisy

Output:

- `local_feat[B,T,L,P,C1]`

### Local Temporal Summary

Do not reduce local outputs with plain last-step selection by default.

Preferred summary:

- gated temporal pooling over `L`

Acceptable fallback:

- mean pooling over `L`

Reason:

- useful spike evidence may be sparse within the raw window
- a learned gate can suppress uninformative spike time steps before global modeling

Output:

- `frame_summary[B,T,P,C1]`

### Global Temporal Mamba

Input:

- `frame_summary[B,T,P,C1]`

Execution:

- fix `(b,p)`
- run Mamba along the frame axis `T`

Recommended block:

- pre-norm
- Mamba over `T`
- residual
- lightweight FFN
- residual

Guidelines:

- start with `1` global layer
- keep the global stage lighter than the local stage
- treat this stage as a temporal enhancer, not a second full backbone

Output:

- `global_feat[B,T,P,C2]`

## Fusion and Write-Back

After global temporal modeling:

1. restore or upsample spike tokens to spatial feature maps
2. encode RGB context with a lightweight `rgb_context_encoder`
3. fuse spike and RGB branches
4. predict conservative residual updates only

Recommended flow:

- `rgb_ctx[B,T,Pr,C]`
- `spike_ctx[B,T,Pr,C]`
- `fused = fusion_body(rgb_ctx + spike_ctx)` or `fusion_body(concat(...))`
- heads:
  - `delta`
  - `gate`
- output:
  - `rgb + alpha * sigmoid(gate) * delta`

This preserves a comparable fusion regime with existing `pase_residual` and collapsed Mamba operators:

- fusion remains early
- output remains RGB-shaped
- the operator remains a conservative residual updater

## Complexity and Budget Control

The redesign should be constrained to a budget that can finish training in the current project window.

### Recommended Initial Budget

- `raw_window_length = 21` or `41`
- spatial stride / patch size around `4`
- local Mamba layers = `1`
- global Mamba layers = `1`
- token width roughly near current Mamba token width
- summary = gated temporal pooling

### Guardrails

1. do not run the dual-scale operator on per-pixel full-resolution tokens
2. do not start with a joint `(T,L)` scan baseline as the main path
3. do not widen both local and global stages aggressively in the first version
4. ensure the operator remains cheaper than a full-resolution all-time token mixer

## Experimental Plan

The first objective is to test whether Mamba's temporal strengths become visible once the operator sees raw spike windows and explicit multi-scale time structure.

### Required Baselines

1. `current mamba collapse + tfp4`
2. `pase_residual + tfp`
3. `pase_residual + raw_window`
4. `dual-scale temporal mamba + raw_window`

### Suggested Evaluation Stages

#### Stage 1: Mamba Recovery Check

Compare:

- current collapsed Mamba
- dual-scale temporal Mamba

Goal:

- verify that the redesign improves over the current underpowered Mamba usage

#### Stage 2: Main Comparison

Compare:

- `pase_residual + raw_window`
- `dual-scale temporal mamba + raw_window`

Goal:

- evaluate whether a better-used Mamba can match or exceed a strong raw-window PASE baseline

#### Stage 3: Ablations

At minimum:

1. remove the global temporal Mamba and keep only the local stage
2. replace gated temporal pooling with mean pooling
3. sweep `raw_window_length` over a small set such as `21` and `41`

These ablations answer:

- how much gain comes from raw-window input itself
- whether the second temporal scale is useful
- how sensitive the operator is to temporal window length

## Risks

1. raw windows may introduce more noise than useful fine-grained signal
2. the global Mamba stage may overlap with temporal modeling already handled later by VRT
3. local temporal summarization may discard too much detail before cross-frame modeling
4. even with a better temporal design, PASE may remain stronger if the task favors local adaptive spike encoding over state-space temporal propagation

## Success Criteria

Use tiered criteria.

### Minimum Success

- dual-scale Mamba clearly improves over current collapsed Mamba

### Moderate Success

- dual-scale Mamba approaches `pase_residual + raw_window`

### Strong Success

- dual-scale Mamba reliably exceeds `pase_residual + raw_window` on at least part of the evaluation suite

This framing ensures the design remains scientifically useful even if it does not immediately become the best operator.

## Prior Art and Positioning

The design direction is supported by several existing lines of work:

- hierarchical local/global state-space modeling as in Hi-Mamba
- scan-path and token-organization sensitivity as emphasized by VMamba
- chunked local/global temporal decomposition ideas analogous to Dual Chunk Attention

This operator should therefore be positioned as:

- a hierarchical Mamba fusion design
- a dual-scale temporal state-space model for spike-guided early fusion

Not as:

- a pure direct replacement for the full VRT temporal backbone
- a strict one-layer long-context Mamba benchmark

## Open Implementation Notes

1. The first version should use a new operator name rather than silently mutating the existing collapsed Mamba path.
2. Dataset support for `raw_window` should be shared with the `pase_residual` raw-window work where possible.
3. Metadata and logging should record:
   - spike representation
   - raw window length
   - local/global layer counts
   - summary type
4. Diagnostics should separately expose local-stage and global-stage norms if practical.
