# Phase-wise Batch Size and GT Size Design (Array Schema)

**Date:** 2026-04-21  
**Branch:** codex/sync  
**Scope:** `main_train_vrt.py`, training option schema under `datasets.train`

---

## Problem

After introducing the phase1 fusion-only fast path, phase1 GPU memory usage is significantly lower. Keeping a single global `datasets.train.dataloader_batch_size` and `datasets.train.gt_size` wastes GPU and time during phase1.

Current behavior uses one static train dataset/dataloader config for the whole run, so phase1 and phase2 cannot use different crop size or batch size.

---

## Goals

1. Allow phase1 and phase2 to use different train `dataloader_batch_size`.
2. Allow phase1 and phase2 to use different train `gt_size`.
3. Switch immediately at `current_step == fix_iter` (no epoch-boundary delay).
4. Keep config readable and near existing keys.
5. Remain backward compatible with existing scalar configs.

---

## Non-goals

1. No phase-specific `netG.img_size` changes.
2. No changes to val/test dataloaders.
3. No changes to loss-phase logic (`is_phase1` definition remains unchanged).

---

## Config Schema

Under `datasets.train`, both keys accept either scalar or 2-element array:

- `dataloader_batch_size`: `int | [int,int]`
- `gt_size`: `int | [int,int]`

Array semantics are fixed and documented inline:

- index `0` = phase1 (`current_step < fix_iter`)
- index `1` = phase2 (`current_step >= fix_iter`)

Example:

```jsonc
"datasets": {
  "train": {
    // 批次大小：支持单值或 [phase1, phase2]
    "dataloader_batch_size": [8, 4],

    // 训练裁剪尺寸：支持单值或 [phase1, phase2]
    "gt_size": [128, 96]
  }
}
```

Backward compatibility:

- Scalar values keep old behavior (same value for both phases).

Validation rules:

- Scalar: must be positive int.
- Array: must be length 2, each element positive int.

---

## Runtime Design

### Phase detection

Reuse existing criterion:

- phase1: `fix_iter > 0 && current_step < fix_iter`
- phase2: otherwise

### Value resolution

Introduce a small resolver utility in `main_train_vrt.py`:

- `resolve_phase_value(value, is_phase1, key_name) -> int`
  - scalar -> returns scalar
  - 2-element list/tuple -> returns index 0 for phase1, 1 for phase2
  - else -> raises clear `ValueError`

### Dataloader lifecycle

1. At training startup, resolve active phase values and build `train_set`/`train_loader` using them.
2. During training loop, track `last_is_phase1`.
3. On crossing boundary (`last_is_phase1 != is_phase1`), immediately rebuild:
   - `train_set`
   - `train_sampler` (if DDP)
   - `train_loader`
4. New loader takes effect from next step immediately.

### DDP behavior

When rebuilding under DDP:

- Recompute per-GPU batch from active global batch size.
- Keep existing sampler policy (`DistributedSampler`, `drop_last=True`, per-rank seed flow).
- If active batch size is smaller than `num_gpu` or not divisible, fail fast with explicit error to avoid silent floor behavior.

---

## Why `netG.img_size` does not need phase split

`gt_size` controls random crop in training dataset.

VRT runtime path handles dynamic input shape with window-based padding/masking in stage forward, so training crop size does not require synchronized phase-wise `netG.img_size` changes.

Keeping `netG.img_size` static avoids unnecessary architecture-level coupling and config complexity.

---

## Observability

On rank0, log once at startup and at phase switch:

- current phase
- active `dataloader_batch_size`
- active `gt_size`
- explicit message when train loader is rebuilt

This makes the switch auditable from logs.

---

## Test Plan

1. **Backward compatibility**
   - Use scalar config; verify no behavior change.

2. **Phase switch correctness**
   - Use arrays, small `fix_iter` (e.g. 10).
   - Verify startup uses phase1 values.
   - Verify exactly at crossing to phase2, loader rebuild log appears and active values change.

3. **DDP validation**
   - Ensure divisibility checks trigger meaningful errors for invalid phase batch values.

4. **Performance sanity**
   - Compare phase1 step time and utilization before/after with larger phase1 batch or gt_size.

---

## Risks and Mitigations

1. **Risk:** Frequent rebuild if phase condition oscillates.
   - **Mitigation:** phase boundary is monotonic (`current_step` only increases), so switch occurs once.

2. **Risk:** Hidden config misuse due to array order confusion.
   - **Mitigation:** inline comment at config keys with fixed `[phase1, phase2]` semantics + strict length/type validation.

3. **Risk:** DDP silent batch truncation.
   - **Mitigation:** explicit divisibility checks before loader build.

---

## Implementation Notes

- Keep changes localized to `main_train_vrt.py` loader construction path.
- Do not refactor unrelated training logic.
- Preserve existing option names and nearby placement for readability.
