# Fusion Parameter Count Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI and helper that reports the complete parameter count of the configured fusion module.

**Architecture:** Add `scripts/analysis/fusion_params.py` with pure helper functions plus a `main()` CLI. The helper counts all parameters under `netG.fusion_adapter`, deduplicating shared parameter objects, and the CLI builds `netG` from an option JSON using existing project construction paths.

**Tech Stack:** Python, PyTorch `nn.Module`, existing `utils.utils_option` and `models.select_network`, pytest.

---

### Task 1: Counting Helper

**Files:**
- Create: `scripts/analysis/fusion_params.py`
- Test: `tests/analysis/test_fusion_params.py`

- [ ] **Step 1: Write failing tests**

Add tests that import `count_fusion_parameters` and `FusionParameterCountError` from `scripts.analysis.fusion_params`. Build dummy models with `fusion_adapter` containing nested `nn.Linear` layers and one shared `nn.Parameter`. Assert the expected total equals the sum of unique parameters, including currently frozen parameters. Add one test where `fusion_enabled=False` and one where `fusion_adapter=None`, both expecting a clear exception.

- [ ] **Step 2: Run tests to verify RED**

Run: `uv run pytest tests/analysis/test_fusion_params.py -q`

Expected: import failure because `scripts.analysis.fusion_params` does not exist yet.

- [ ] **Step 3: Implement minimal helper**

Create `scripts/analysis/fusion_params.py` with:

```python
class FusionParameterCountError(RuntimeError):
    pass


def _unique_parameters(module):
    seen = set()
    for param in module.parameters():
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        yield param


def count_fusion_parameters(net):
    if not bool(getattr(net, "fusion_enabled", False)):
        raise FusionParameterCountError("Fusion is not enabled on this model.")
    fusion_module = getattr(net, "fusion_adapter", None)
    if fusion_module is None:
        raise FusionParameterCountError("Model has no fusion_adapter to count.")
    return sum(param.numel() for param in _unique_parameters(fusion_module))
```

- [ ] **Step 4: Run tests to verify GREEN**

Run: `uv run pytest tests/analysis/test_fusion_params.py -q`

Expected: all tests pass.

### Task 2: CLI

**Files:**
- Modify: `scripts/analysis/fusion_params.py`
- Test: `tests/analysis/test_fusion_params.py`

- [ ] **Step 1: Write failing CLI formatting tests**

Add tests for `format_parameter_count(1234567)` returning `1,234,567 (1.235 M)` and for `build_arg_parser()` accepting an option path.

- [ ] **Step 2: Run tests to verify RED**

Run: `uv run pytest tests/analysis/test_fusion_params.py -q`

Expected: failure because formatting/parser helpers do not exist.

- [ ] **Step 3: Implement CLI helpers and main**

Add:

```python
def format_parameter_count(count):
    return f"{count:,} ({count / 1_000_000:.3f} M)"
```

Add an argparse parser with positional `opt` and a `main()` that parses the option, builds `netG`, counts fusion parameters, prints `Fusion parameters: ...`, and returns exit code 0. Catch `FusionParameterCountError` to print a readable error and return exit code 2.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/analysis/test_fusion_params.py -q`

Expected: all tests pass.

### Task 3: Smoke Run

**Files:**
- Modify: none expected unless smoke exposes a missing import or option parsing issue.

- [ ] **Step 1: Run CLI against local/server option**

Run: `uv run python scripts/analysis/fusion_params.py options/gopro_rgbspike_server.json`

Expected: prints one `Fusion parameters:` line. If optional runtime dependencies prevent construction, document the exact blocker and keep unit tests passing.

- [ ] **Step 2: Run relevant tests**

Run: `uv run pytest tests/analysis/test_fusion_params.py -q`

Expected: all tests pass.
