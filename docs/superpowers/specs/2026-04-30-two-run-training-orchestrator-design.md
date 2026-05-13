# Two-Run Training Orchestrator Design

**Date:** 2026-04-30
**Status:** Approved design, pending implementation

## Summary

Add a true two-run training orchestrator to `main_train_vrt.py` so a single training launch can execute:

1. phase 1: fusion-only warmup
2. phase 2: fresh fine-tuning from the phase-1 boundary checkpoint with a new optimizer and a new scheduler

The user experience must remain:

- one launch command
- one experiment directory
- one TensorBoard run
- one wandb run
- one SwanLab run
- one continuous external step axis

The optimization semantics must change at the phase boundary:

- phase 2 must inherit model weights from phase 1
- phase 2 must not inherit phase-1 optimizer state
- phase 2 must not inherit phase-1 scheduler progress
- phase 2 must not inherit phase-1 AMP scaler state

This design replaces the current pseudo-two-stage behavior where `fix_iter` only changes trainability inside one continuous optimizer/scheduler trajectory.

## Problem

The current snapshot-style training config uses one continuous optimizer and one continuous scheduler across the entire run.

In the present setup:

- phase 1 trains fusion while backbone, LoRA, and flow-alignment adaptation are constrained
- phase 2 starts later by changing trainability inside the same process
- the scheduler keeps decaying from the original step 0 rather than restarting for phase 2
- optimizer moments are also inherited from the phase-1 optimization regime

This creates an optimization mismatch:

- phase 2 is semantically a new optimization problem
- but it is forced to continue phase-1 optimizer history and scheduler progress

For the current formal training pattern, this is especially undesirable because:

- phase 1 and phase 2 optimize different parameter sets
- phase 2 introduces LoRA and low-LR flow/alignment adaptation
- phase 2 should have its own learning-rate trajectory

## Goals

1. Preserve single-launch training UX.
2. Preserve one experiment directory and one external tracking run.
3. Make phase 2 a true fresh optimization stage.
4. Keep phase-1 and phase-2 configuration in one human-managed JSON file.
5. Make phase boundaries explicit, inspectable, and resumable.
6. Keep the design reusable for future fusion/operator experiments.

## Non-Goals

1. Do not add a second standalone training entrypoint.
2. Do not reset the scheduler in-place inside one training loop.
3. Do not preserve phase-1 optimizer or scaler state into phase 2.
4. Do not split one experiment into two separate TensorBoard or tracking runs.
5. Do not require users to maintain two large duplicated JSON configs.

## Rationale

### Research and optimization rationale

This design is motivated by common stage-wise optimization practice rather than by a claim that all staged training must always use separate optimizers.

Relevant precedent:

- ULMFiT uses gradual unfreezing and stage-sensitive fine-tuning behavior rather than treating all phases as one unchanged optimization regime.
- Faster R-CNN explicitly uses alternating optimization because jointly changing tightly coupled components at once is not assumed to be the best optimization path.
- LoRA+ shows that even within adapter fine-tuning, different low-rank components can benefit from different optimization treatment instead of blindly sharing one identical trajectory.
- SGDR establishes restart-style learning-rate behavior as a valid optimization tool rather than an abnormal intervention.

For this codebase, the practical implication is:

- phase 1 and phase 2 do not merely differ in loss weight
- they differ in trainable parameter set, optimization target, and desired LR schedule

That makes a true phase boundary with a fresh optimizer and scheduler the cleaner design.

### Repository-specific rationale

This repository already explored a true two-run training idea in earlier design work, but the implemented path later converged on a single-config continuous scheduler approach.

That continuous approach is convenient for a single-loop implementation, but it does not match the semantics desired for:

- fusion-only warmup
- fresh LoRA adaptation
- fresh SCFlow/DCN adaptation

The orchestrator proposed here restores true two-run semantics without giving up the convenience of single-launch training.

## High-Level Design

Use one base config file with a new `train.two_run` section.

At runtime, the trainer does the following:

1. parse the base config
2. resolve `phase1_opt` from the base config plus `train.two_run.phase1`
3. resolve `phase2_opt` from the base config plus `train.two_run.phase2`
4. initialize shared runtime objects once
5. run phase 1 to completion
6. save a boundary checkpoint
7. tear down phase-1 training objects
8. rebuild phase-2 training objects from fresh config and fresh optimization state
9. continue logging to the same external run with continuous `global_step`

This creates two optimization runs inside one process-level experiment lifecycle.

## User Experience

The intended user workflow remains:

- edit one snapshot config
- launch one command
- monitor one experiment directory
- inspect one continuous TensorBoard / wandb / SwanLab curve

The user should not need to:

- manually launch phase 2
- manually point phase 2 to a phase-1 checkpoint
- manually manage two separate task names
- manually merge metrics from two runs

## Configuration Model

### Base principle

The source of truth remains a single config file.

Shared settings stay in their current places:

- `datasets`
- `netG`
- `path`
- `val`
- `logging`
- phase-agnostic parts of `train`

Phase-specific differences are declared under:

```json
"train": {
  "two_run": {
    "enable": true,
    "phase1": { ...overrides... },
    "phase2": { ...overrides... }
  }
}
```

### Why override blocks are preferred

This is preferred over two full nested configs because:

- it keeps one experiment definition
- it avoids duplicated large JSON structures
- it makes phase differences obvious in review
- it reduces config drift between phases

### Phase-specific fields

Typical phase-specific fields include:

- `total_iter`
- `use_lora`
- `phase2_lora_mode`
- `fix_iter`
- `fix_keys`
- `fix_lr_mul`
- `trainable_extra_keys`
- `G_optimizer_lr`
- `G_optimizer_reuse`
- `G_scheduler_type`
- `G_scheduler_periods`
- `G_scheduler_milestones`
- `checkpoint_test`
- `checkpoint_save`

### Example shape

```json
"train": {
  "freeze_backbone": true,
  "partial_load": true,
  "G_optimizer_type": "adam",
  "G_optimizer_betas": [0.9, 0.99],
  "two_run": {
    "enable": true,
    "phase1": {
      "total_iter": 4000,
      "use_lora": false,
      "fix_iter": 0,
      "fix_keys": [],
      "G_optimizer_lr": 4e-4,
      "G_scheduler_type": "CosineAnnealingWarmRestarts",
      "G_scheduler_periods": 4000,
      "checkpoint_test": [4000],
      "checkpoint_save": 2000
    },
    "phase2": {
      "total_iter": 6000,
      "use_lora": true,
      "phase2_lora_mode": false,
      "trainable_extra_keys": ["spynet", "pa_deform"],
      "fix_iter": 0,
      "fix_keys": ["spynet", "pa_deform"],
      "fix_lr_mul": 0.1,
      "G_optimizer_lr": 2e-4,
      "G_optimizer_reuse": false,
      "G_scheduler_type": "CosineAnnealingWarmRestarts",
      "G_scheduler_periods": 6000,
      "checkpoint_test": [2000, 4000, 6000],
      "checkpoint_save": 2000
    }
  }
}
```

## Config Composition Rules

### Merge behavior

Resolved phase configs are built by:

1. deep-copying the base config
2. recursively merging the selected phase override block

Rules:

- scalars overwrite
- dictionaries merge recursively
- lists replace the full old list
- `null` is a valid explicit override

### Required validation

If `train.two_run.enable=true`:

1. `phase1.total_iter` must exist and be positive
2. `phase2.total_iter` must exist and be positive
3. phase-specific scheduler fields must be internally valid
4. phase-2 config must not attempt to preserve phase-1 optimizer reuse semantics

### Forced phase-2 runtime injection

At runtime, the orchestrator must force:

- `phase2_opt['path']['pretrained_netG'] = phase1_final_G`
- `phase2_opt['path']['pretrained_netE'] = phase1_final_E` when EMA exists
- `phase2_opt['path']['pretrained_optimizerG'] = None`
- `phase2_opt['train']['G_optimizer_reuse'] = False`

If the user explicitly sets conflicting values in the phase-2 override block, the trainer should raise a clear error instead of accepting them.

### Resolved-config artifacts

For observability, each run should save:

- one base config snapshot
- one resolved phase-1 config snapshot
- one resolved phase-2 config snapshot

Suggested filenames:

- `options/<timestamp>_base.json`
- `options/<timestamp>_phase1_resolved.json`
- `options/<timestamp>_phase2_resolved.json`

## Runtime Architecture

### Shared experiment lifetime

One process launch owns one experiment lifetime.

Objects created once per experiment:

- distributed-process initialization
- experiment directory creation
- Python file logger
- TensorBoard writer
- wandb run
- SwanLab run

Objects created once per phase:

- model instance
- optimizer
- scheduler
- AMP scaler
- train dataset
- train dataloader
- train sampler
- phase profiler/timer state

### Structure

Recommended top-level structure:

- `run_experiment(opt)`
- `resolve_two_run_phase_opts(opt)`
- `run_phase(phase_opt, shared_runtime, phase_name, global_step_offset, resume_state)`
- `finalize_phase(...)`
- `load_or_initialize_two_run_state(...)`
- `persist_two_run_state(...)`

The intent is to isolate:

- experiment-wide lifecycle
- phase-local training lifecycle
- resume/manifest state

## Phase Boundary Semantics

### State that must be inherited into phase 2

- `netG` weights from phase-1 final checkpoint
- `netE` weights from phase-1 final checkpoint when EMA exists
- experiment directory
- logger and tracking run identity
- continuous external `global_step`

### State that must be rebuilt for phase 2

- model instance
- optimizer
- scheduler
- AMP scaler
- dataloader and sampler
- phase-local timer/profiler state
- phase-local step counter

### State that must not be inherited into phase 2

- optimizer moments
- scheduler progression
- scaler state
- phase-1 trainability state

This is the central semantic rule of the design.

## Step Semantics

Two step axes are required.

### `phase_step`

Used for:

- optimizer stepping
- scheduler stepping
- phase-local termination
- phase-local validation triggers
- phase-local save cadence

Phase-local ranges:

- phase 1: `1..phase1_total_iter`
- phase 2: `1..phase2_total_iter`

### `global_step`

Used for:

- TensorBoard logging
- wandb logging
- SwanLab logging
- human-facing training logs
- checkpoint naming

Global ranges:

- phase 1: `1..phase1_total_iter`
- phase 2: `phase1_total_iter + 1 .. phase1_total_iter + phase2_total_iter`

### Why the split is necessary

This is what lets:

- phase 2 scheduler restart from its own step 1
- external curves remain continuous

## Training Lifecycle

### 1. Startup

At launch:

1. parse base config
2. initialize distributed environment
3. create shared logging/tracking runtime
4. resolve `phase1_opt` and `phase2_opt`
5. save base and resolved phase configs
6. load or initialize the two-run manifest

### 2. Phase 1

Phase 1 should:

1. build fresh phase-1 training objects
2. train until `phase1.total_iter`
3. save a guaranteed phase boundary checkpoint
4. run boundary validation
5. mark phase 1 completed in the manifest

### 3. Boundary transition

Between phases:

1. synchronize ranks
2. release phase-1 training objects
3. run `gc.collect()`
4. clear CUDA cache where appropriate
5. inject phase-1 final checkpoints into resolved `phase2_opt`
6. mark phase 2 as started in the manifest

### 4. Phase 2

Phase 2 should:

1. build fresh phase-2 training objects
2. load phase-1 model weights only
3. train with fresh optimizer/scheduler/scaler state
4. continue external logging on the same `global_step` axis
5. close writers and tracking runs only after final completion

## Checkpoint Strategy

### Naming

Checkpoint files should use `global_step` in filenames, not `phase_step`.

Examples:

- phase-1 final `4000_G.pth`
- phase-2 mid-run `6000_G.pth`
- phase-2 final `10000_G.pth`

This avoids filename collision and keeps filenames aligned with the external metric axis.

### Why global-step filenames are preferred

Benefits:

- unique filenames across both phases
- compatibility with the current `find_last_checkpoint()` style
- easy correlation with TensorBoard/tracking charts
- no ambiguity when inspecting model directories

## Resume and Failure Recovery

### Manifest

Add a small manifest file in the experiment directory:

- `experiments/<task>/two_run_state.json`

Suggested fields:

- `two_run_enabled`
- `current_phase`
- `phase1_total_iter`
- `phase2_total_iter`
- `phase1_completed`
- `phase1_final_G`
- `phase1_final_E`
- `phase2_started`
- `global_step_offset`
- `last_successful_phase_step`
- `last_successful_global_step`

### Resume rules

1. If no manifest exists:
   - start a fresh experiment at phase 1

2. If the manifest exists and `phase1_completed=false`:
   - resume phase 1 only

3. If the manifest exists, `phase1_completed=true`, and `phase2_started=false`:
   - start phase 2 from the phase-1 final checkpoint
   - use fresh optimizer/scheduler/scaler state

4. If the manifest exists and `phase2_started=true`:
   - resume phase 2 from its own latest checkpoints

### Important safety rule

Do not infer current phase only from the highest checkpoint step.

The manifest is the authoritative source for phase identity.

This avoids fragile heuristics and prevents accidental re-entry into the wrong phase.

## Logging and Tracking

### One external run

TensorBoard, wandb, and SwanLab must stay attached to the same experiment run for both phases.

That means:

- do not close the writer/run at the phase-1 boundary
- do not create a second tracking run for phase 2

### Logging behavior

Recommended log messages:

- resolved phase summaries at startup
- phase transition announcement
- phase-1 boundary checkpoint path
- phase-2 fresh optimizer/scheduler confirmation
- resume source details

Examples:

- `[TWO_RUN] phase1 resolved total_iter=4000 lr=4.000e-4 scheduler=CosineAnnealingWarmRestarts`
- `[TWO_RUN] phase2 resolved total_iter=6000 lr=2.000e-4 scheduler=CosineAnnealingWarmRestarts`
- `[TWO_RUN] phase2 will load G from phase1_final_G with fresh optimizer/scheduler`

## Compatibility with Existing Training Semantics

### When `two_run.enable=false`

The current training path must remain unchanged.

This preserves backward compatibility for:

- existing snapshot configs
- local debug configs
- continuous single-loop experiments

### When `two_run.enable=true`

The trainer must not rely on single-loop `fix_iter` semantics for phase switching.

In this mode:

- phase 1 and phase 2 are independent resolved configs
- phase switching is owned by the orchestrator
- not by `ModelVRT.optimize_parameters()` trainability branching

## File Map

### `main_train_vrt.py`

Primary implementation site.

Responsibilities:

- add two-run config resolution
- add shared experiment runtime
- add per-phase runner
- add boundary transition logic
- add manifest lifecycle
- route external logging through `global_step`

### `utils/utils_option.py`

Responsibilities:

- add `two_run` defaults and validation helpers
- support resolved-phase config dumping if implemented there

### `utils/utils_logger.py`

Expected changes should be small.

Responsibilities:

- preserve one long-lived logger/tracking lifecycle across phases
- avoid phase-boundary close/reopen churn

### `models/model_plain.py`

Likely no structural redesign needed.

Main requirement:

- phase-2 fresh-start behavior must work cleanly when `pretrained_optimizerG=None` and `G_optimizer_reuse=false`

### `models/model_vrt.py`

Main requirement:

- two-run mode should not depend on in-loop pseudo-phase switching for the formal phase boundary

It may still keep existing single-loop functionality for backward compatibility when `two_run.enable=false`.

## Testing Strategy

### Unit tests

1. config resolution
   - base + phase1 override merge
   - base + phase2 override merge
   - list replacement semantics

2. validation
   - missing phase `total_iter`
   - forbidden phase-2 optimizer reuse
   - conflicting forced checkpoint fields

3. manifest behavior
   - fresh startup
   - resume phase 1
   - phase-1-complete to phase-2-start transition
   - resume phase 2

4. step mapping
   - `phase_step` resets
   - `global_step` continues

### Integration tests

1. one short two-run smoke test
   - phase 1: a few iterations
   - phase 2: a few iterations
   - assert one continuous checkpoint namespace
   - assert fresh phase-2 optimizer construction

2. resume smoke tests
   - interrupt after phase 1
   - interrupt during phase 2
   - confirm correct recovery path

3. logging smoke test
   - one TensorBoard writer lifecycle
   - one tracking-run lifecycle

## Risks

### Risk: phase-2 resume accidentally loads phase-1 optimizer state

Mitigation:

- explicitly force `pretrained_optimizerG=None`
- explicitly force `G_optimizer_reuse=false`
- validate against conflicting user config

### Risk: step-axis confusion causes duplicate logs or checkpoint collisions

Mitigation:

- make `phase_step` and `global_step` separate variables
- use `global_step` for filenames and external logging only

### Risk: large `main_train_vrt.py` becomes harder to maintain

Mitigation:

- refactor into small helper functions rather than appending more inline logic

### Risk: resume heuristics become brittle

Mitigation:

- use an explicit manifest as the source of truth

## Out of Scope

1. Reworking all existing single-loop two-stage configs.
2. Designing a generic N-phase training system.
3. Adding a third optional finishing phase.
4. Adding a separate launcher/orchestrator script outside `main_train_vrt.py`.
5. Changing fusion operator semantics as part of this work.

## Final Decision

Implement a true process-internal two-run orchestrator with:

- one base config file
- one `train.two_run` override block
- one experiment directory
- one continuous external metric axis
- two independent optimization phases

This is the cleanest way to recover true staged-training semantics without giving up the operational convenience of the current single-launch workflow.
