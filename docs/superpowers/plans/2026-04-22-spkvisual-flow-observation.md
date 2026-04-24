# spkvisual Flow Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a server-runnable `spkvisual` flow observation tool that compares SpyNet and SCFlow flow statistics and persists all conclusions as artifacts.

**Architecture:** Keep the tool standalone in `spkvisual/observe_flow_stats.py`, with pure helper functions for statistics and artifact writing so they can be unit-tested locally. The script directly reads RGB frame paths and `encoding25` artifacts from the selected dataset config instead of using the randomized training dataset path.

**Tech Stack:** Python, PyTorch, OpenCV, NumPy, Matplotlib, CSV/JSON, pytest.

---

### Task 1: Helper Tests

**Files:**
- Create: `tests/spkvisual/test_observe_flow_stats.py`
- Create: `spkvisual/observe_flow_stats.py`

- [ ] **Step 1: Write failing tests** for flow statistics, persisted summary structure, and subframe selection.
- [ ] **Step 2: Run** `pytest tests/spkvisual/test_observe_flow_stats.py -q` and confirm it fails because `spkvisual.observe_flow_stats` does not exist.
- [ ] **Step 3: Implement pure helpers** in `spkvisual/observe_flow_stats.py`: JSON comment stripping, tensor summary, active mask summary, difference summary, subframe selection, and artifact writers.
- [ ] **Step 4: Re-run** `pytest tests/spkvisual/test_observe_flow_stats.py -q` and confirm it passes.

### Task 2: CLI Runtime

**Files:**
- Modify: `spkvisual/observe_flow_stats.py`

- [ ] **Step 1: Add argparse CLI** with options for `--opt`, `--dataset`, `--num-pairs`, `--start-index`, `--device`, `--spynet-ckpt`, `--scflow-ckpt`, `--subframe`, `--low-flow-threshold`, `--active-threshold`, and `--out`.
- [ ] **Step 2: Add dataset resolution** from `meta_info_file`, RGB LQ root, spike flow root, filename template, extension, spatial shape, `dt`, artifact format, and subframe count.
- [ ] **Step 3: Add model execution** using `create_optical_flow("spynet", ...)` and `create_optical_flow("scflow", ..., dt=...)` under `torch.no_grad()`.
- [ ] **Step 4: Add artifact generation** for `summary.json`, `per_pair.csv`, and three histogram PNGs.

### Task 3: Verification and Git

**Files:**
- Modify: `spkvisual/observe_flow_stats.py`
- Create: `tests/spkvisual/test_observe_flow_stats.py`
- Create: `docs/superpowers/specs/2026-04-22-spkvisual-flow-observation-design.md`
- Create: `docs/superpowers/plans/2026-04-22-spkvisual-flow-observation.md`

- [ ] **Step 1: Run** `pytest tests/spkvisual/test_observe_flow_stats.py -q`.
- [ ] **Step 2: Run** `python spkvisual/observe_flow_stats.py --help`.
- [ ] **Step 3: Inspect** `git diff --check`.
- [ ] **Step 4: Commit** with `feat(spkvisual): add flow observation tool`.
- [ ] **Step 5: Push** current branch and report the server command for the user to run.
