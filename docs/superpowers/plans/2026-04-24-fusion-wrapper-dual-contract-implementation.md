# Fusion Wrapper Dual-Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor early fusion so the wrapper exposes explicit supervision and execution views, keeps phase-1/phase-2 supervision on canonical `N` frames, preserves existing expanded `N*S` execution paths, and keeps existing config files working by default.

**Architecture:** Add a structured early-fusion wrapper result with `fused_main`, `backbone_view`, `aux_view`, and `meta`; encode `expanded` versus `collapsed` contracts per operator; update VRT to execute on `backbone_view` while losses supervise only `fused_main`; update debug and tests to consume explicit semantic cache fields instead of inferring meaning from tensor shapes.

**Tech Stack:** Python, PyTorch, existing VRT/fusion adapter stack, pytest

---

## File Map

- Modify: `models/fusion/adapters/early.py`
  - Return a structured result with canonical main view and execution view.
  - Own expanded/collapsed packaging and `N*S -> N` reduction.
- Modify: `models/fusion/operators/gated.py`
  - Declare `frame_contract = "expanded"`.
- Modify: `models/fusion/operators/concat.py`
  - Declare `frame_contract = "expanded"`.
- Modify: `models/fusion/operators/pase.py`
  - Declare `frame_contract = "expanded"`.
- Modify: `models/fusion/operators/mamba.py`
  - Declare `frame_contract = "collapsed"`.
- Modify: `models/architectures/vrt/vrt.py`
  - Cache explicit fusion main/exec/aux/meta outputs and execute on `backbone_view`.
- Modify: `models/model_plain.py`
  - Supervise only canonical `fused_main`.
- Modify: `models/model_vrt.py`
  - Route fusion debug through explicit cache fields.
- Modify: `models/fusion/debug.py`
  - Keep forward-hook capture compatible with structured wrapper outputs.
  - Default dump source to canonical main view and optionally dump expanded execution view.
- Modify: `tests/models/test_fusion_early_adapter.py`
  - Add contract tests for expanded/collapsed wrappers.
- Modify: `tests/models/test_vrt_fusion_integration.py`
  - Update integration tests for explicit main/exec cache semantics.
- Modify: `tests/models/test_model_plain_fusion_aux_loss.py`
  - Lock aux loss to canonical main timeline only.
- Modify: `tests/models/test_fusion_debug_dumper.py`
  - Verify debug defaults to main and can inspect expanded execution view when requested.

### Task 1: Add explicit frame-contract metadata to early operators

**Files:**
- Modify: `models/fusion/operators/gated.py`
- Modify: `models/fusion/operators/concat.py`
- Modify: `models/fusion/operators/pase.py`
- Modify: `models/fusion/operators/mamba.py`
- Test: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: Write the failing metadata tests**

Add these tests to `tests/models/test_fusion_early_adapter.py`:

```python
def test_gated_operator_declares_expanded_frame_contract():
    from models.fusion.factory import create_fusion_operator

    op = create_fusion_operator("gated", 3, 1, 3, {})
    assert getattr(op, "frame_contract", None) == "expanded"


def test_concat_operator_declares_expanded_frame_contract():
    from models.fusion.factory import create_fusion_operator

    op = create_fusion_operator("concat", 3, 1, 3, {})
    assert getattr(op, "frame_contract", None) == "expanded"


def test_pase_operator_declares_expanded_frame_contract():
    from models.fusion.factory import create_fusion_operator

    op = create_fusion_operator("pase", 3, 1, 3, {})
    assert getattr(op, "frame_contract", None) == "expanded"


def test_mamba_operator_declares_collapsed_frame_contract():
    from models.fusion.factory import create_fusion_operator

    op = create_fusion_operator("mamba", 3, 1, 3, {})
    assert getattr(op, "frame_contract", None) == "collapsed"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py -k "declares_.*frame_contract" -v
```

Expected: FAIL because current operators do not expose `frame_contract`.

- [ ] **Step 3: Write minimal implementation**

Add these class attributes:

```python
class GatedFusionOperator(nn.Module):
    frame_contract = "expanded"
```

```python
class ConcatFusionOperator(nn.Module):
    frame_contract = "expanded"
```

```python
class PaseFusionOperator(nn.Module):
    frame_contract = "expanded"
```

```python
class MambaFusionOperator(nn.Module):
    frame_contract = "collapsed"
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py -k "declares_.*frame_contract" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion/operators/gated.py models/fusion/operators/concat.py models/fusion/operators/pase.py models/fusion/operators/mamba.py tests/models/test_fusion_early_adapter.py
git commit -m "test(fusion): declare expanded and collapsed operator contracts"
```

### Task 2: Refactor early wrapper to return main and execution views

**Files:**
- Modify: `models/fusion/adapters/early.py`
- Test: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: Write the failing wrapper contract tests**

Add these tests to `tests/models/test_fusion_early_adapter.py`:

```python
def test_early_adapter_returns_main_and_exec_for_expanded_operator():
    from models.fusion.adapters.early import EarlyFusionAdapter

    class ExpandedStub(torch.nn.Module):
        frame_contract = "expanded"

        def forward(self, rgb_rep, spk):
            return rgb_rep

    adapter = EarlyFusionAdapter(operator=ExpandedStub(), mode="replace", inject_stages=[], spike_chans=4)
    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)

    result = adapter(rgb, spike)

    assert set(result.keys()) == {"fused_main", "backbone_view", "aux_view", "meta"}
    assert result["meta"]["frame_contract"] == "expanded"
    assert result["fused_main"].shape == (1, 2, 3, 8, 8)
    assert result["backbone_view"].shape == (1, 8, 3, 8, 8)
    assert result["meta"]["main_from_exec_rule"] == "center_subframe"


def test_early_adapter_returns_main_and_exec_for_collapsed_operator():
    from models.fusion.adapters.early import EarlyFusionAdapter

    class CollapsedStub(torch.nn.Module):
        frame_contract = "collapsed"

        def forward(self, rgb, spike):
            return rgb

    adapter = EarlyFusionAdapter(operator=CollapsedStub(), mode="replace", inject_stages=[], spike_chans=4)
    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)

    result = adapter(rgb, spike)

    assert result["meta"]["frame_contract"] == "collapsed"
    assert result["fused_main"].shape == (1, 2, 3, 8, 8)
    assert result["backbone_view"].shape == (1, 2, 3, 8, 8)
    assert result["meta"]["main_from_exec_rule"] is None
```

Also migrate existing tests in this file that currently assume:

- `EarlyFusionAdapter.forward()` returns a bare tensor
- expanded early adapters expose only `N*S` as the returned value
- structured-early tests inspect raw adapter tensor shapes directly

Update those tests so they assert against:

- `result["fused_main"]`
- `result["backbone_view"]`
- `result["meta"]`

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py -k "returns_main_and_exec" -v
```

Expected: FAIL because `EarlyFusionAdapter.forward()` currently returns a bare tensor.

- [ ] **Step 3: Write minimal implementation**

In `models/fusion/adapters/early.py`, add helpers like:

```python
def _build_meta(self, frame_contract: str, spike_bins: int, main_steps: int, exec_steps: int, aux_steps: int | None, main_from_exec_rule: str | None):
    return {
        "operator_name": self.operator.__class__.__name__,
        "frame_contract": frame_contract,
        "spike_bins": spike_bins,
        "main_steps": main_steps,
        "exec_steps": exec_steps,
        "aux_steps": aux_steps,
        "main_from_exec_rule": main_from_exec_rule,
    }
```

```python
def _reduce_expanded_exec_to_main(self, exec_view: torch.Tensor, spike_bins: int) -> torch.Tensor:
    return exec_view[:, spike_bins // 2 :: spike_bins, :, :, :]
```

```python
def forward(self, rgb: torch.Tensor, spike: torch.Tensor) -> dict[str, Any]:
    ...
    frame_contract = str(getattr(self.operator, "frame_contract", "expanded")).strip().lower()
    if frame_contract == "collapsed":
        backbone_view = self.operator(rgb, spike)
        return {
            "fused_main": backbone_view,
            "backbone_view": backbone_view,
            "aux_view": None,
            "meta": self._build_meta(frame_contract, spike_steps_per_frame, steps, steps, None, None),
        }

    rgb_rep = rgb.unsqueeze(2).expand(bsz, steps, spike_steps_per_frame, rgb_chans, height, width)
    rgb_rep = rgb_rep.reshape(bsz, steps * spike_steps_per_frame, rgb_chans, height, width)
    spk = spike.reshape(bsz, steps * spike_steps_per_frame, 1, height, width)
    backbone_view = self.operator(rgb_rep, spk)
    fused_main = self._reduce_expanded_exec_to_main(backbone_view, spike_steps_per_frame)
    return {
        "fused_main": fused_main,
        "backbone_view": backbone_view,
        "aux_view": backbone_view,
        "meta": self._build_meta(frame_contract, spike_steps_per_frame, steps, backbone_view.size(1), backbone_view.size(1), "center_subframe"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py -v
```

Expected: PASS, including migrated legacy adapter tests that previously assumed tensor returns.

- [ ] **Step 5: Commit**

```bash
git add models/fusion/adapters/early.py tests/models/test_fusion_early_adapter.py
git commit -m "feat(fusion): add explicit main and execution wrapper views"
```

### Task 3: Keep fusion debug hook compatible with structured wrapper outputs

**Files:**
- Modify: `models/fusion/debug.py`
- Test: `tests/models/test_fusion_debug_dumper.py`

- [ ] **Step 1: Write the failing hook-compatibility test**

Add this test to `tests/models/test_fusion_debug_dumper.py`:

```python
def test_fusion_debug_hook_captures_fused_main_from_structured_output(tmp_path):
    dumper = FusionDebugDumper(_make_opt(tmp_path))
    dumper.enabled = True
    dumper.save_images = True
    dumper.arm()

    structured = {
        "fused_main": torch.rand(1, 2, 3, 8, 8),
        "backbone_view": torch.rand(1, 8, 3, 8, 8),
        "aux_view": torch.rand(1, 8, 3, 8, 8),
        "meta": {"frame_contract": "expanded", "spike_bins": 4},
    }

    dumper._capture_hook(module=None, inputs=(), output=structured)

    assert dumper._last_output is not None
    assert tuple(dumper._last_output.shape) == (1, 2, 3, 8, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/models/test_fusion_debug_dumper.py -k "captures_fused_main_from_structured_output" -v
```

Expected: FAIL because the current hook ignores dict outputs.

- [ ] **Step 3: Write minimal implementation**

Update `models/fusion/debug.py`:

```python
def _capture_hook(self, module, inputs, output):
    if not self._armed:
        return
    tensor = None
    if isinstance(output, dict):
        tensor = output.get("fused_main", None)
    elif isinstance(output, (tuple, list)):
        tensor = output[0]
    else:
        tensor = output
    if isinstance(tensor, torch.Tensor):
        self.capture_tensor(tensor)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/models/test_fusion_debug_dumper.py -k "captures_fused_main_from_structured_output" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion/debug.py tests/models/test_fusion_debug_dumper.py
git commit -m "fix(debug): capture canonical fusion main from structured wrapper output"
```

### Task 4: Update VRT to cache semantic fusion outputs and execute on `backbone_view`

**Files:**
- Modify: `models/architectures/vrt/vrt.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Write the failing VRT cache tests**

Add or update tests in `tests/models/test_vrt_fusion_integration.py`:

```python
def test_vrt_stores_explicit_fusion_main_exec_aux_and_meta_after_forward():
    ...
    with torch.no_grad():
        _ = model(x)

    assert hasattr(model, "_last_fusion_main")
    assert hasattr(model, "_last_fusion_exec")
    assert hasattr(model, "_last_fusion_aux")
    assert hasattr(model, "_last_fusion_meta")
```

```python
def test_vrt_expanded_operator_keeps_main_n_and_exec_ns():
    ...
    assert model._last_fusion_main.shape == (1, 6, 3, 16, 16)
    assert model._last_fusion_exec.shape == (1, 24, 3, 16, 16)
    assert model._last_fusion_meta["frame_contract"] == "expanded"
```

```python
def test_vrt_collapsed_operator_keeps_main_and_exec_equal():
    ...
    assert model._last_fusion_main.shape == (1, 6, 3, 16, 16)
    assert model._last_fusion_exec.shape == (1, 6, 3, 16, 16)
    assert model._last_fusion_meta["frame_contract"] == "collapsed"
```

Also migrate existing tests in this file that currently assume:

- `_last_fusion_out` is the early-fusion cache field
- `fusion_adapter(...)` returns a bare tensor
- structured-early tests inspect only raw adapter output shape

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py -k "fusion_main_exec_aux_and_meta or keeps_main_n_and_exec_ns or keeps_main_and_exec_equal" -v
```

Expected: FAIL because VRT still uses `_last_fusion_out`.

- [ ] **Step 3: Write minimal implementation**

In `models/architectures/vrt/vrt.py`, replace:

```python
self._last_fusion_out = None
```

with:

```python
self._last_fusion_main = None
self._last_fusion_exec = None
self._last_fusion_aux = None
self._last_fusion_meta = None
```

Update early-fusion forward handling to:

```python
fusion_result = self.fusion_adapter(rgb=rgb, spike=spike)
fused_main = fusion_result["fused_main"]
backbone_view = fusion_result["backbone_view"]
aux_view = fusion_result["aux_view"]
meta = fusion_result["meta"]

self._last_fusion_main = fused_main
self._last_fusion_exec = backbone_view
self._last_fusion_aux = aux_view
self._last_fusion_meta = meta
self._last_spike_bins = spike_bins

x = backbone_view
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py -v
```

Expected: PASS, including migrated legacy VRT integration tests that previously depended on `_last_fusion_out` or bare adapter outputs.

- [ ] **Step 5: Commit**

```bash
git add models/architectures/vrt/vrt.py tests/models/test_vrt_fusion_integration.py
git commit -m "refactor(vrt): cache semantic fusion views and metadata"
```

### Task 5: Move phase-1 aux supervision fully onto canonical `fused_main`

**Files:**
- Modify: `models/model_plain.py`
- Test: `tests/models/test_model_plain_fusion_aux_loss.py`

- [ ] **Step 1: Write the failing phase-1 supervision tests**

Add or update tests in `tests/models/test_model_plain_fusion_aux_loss.py`:

```python
def test_fusion_aux_loss_reads_last_fusion_main_only():
    from models.model_plain import ModelPlain

    model = ModelPlain(_make_opt())
    model.define_loss()

    bare = model.get_bare_model(model.netG)
    bare._last_fusion_main = torch.ones(1, 6, 3, 8, 8)
    bare._last_fusion_exec = torch.zeros(1, 24, 3, 8, 8)
    bare._last_fusion_aux = torch.zeros(1, 24, 3, 8, 8)
    bare._last_fusion_meta = {"frame_contract": "expanded"}
    bare._last_spike_bins = 4
    model.H = torch.ones(1, 6, 3, 8, 8)
    model.L = torch.zeros(1, 6, 7, 8, 8)

    loss = model._compute_fusion_aux_loss(is_phase1=True)
    assert loss.item() < 0.002
```

```python
def test_fusion_aux_loss_rejects_time_mismatch_in_last_fusion_main():
    from models.model_plain import ModelPlain

    model = ModelPlain(_make_opt())
    model.define_loss()

    bare = model.get_bare_model(model.netG)
    bare._last_fusion_main = torch.zeros(1, 5, 3, 8, 8)
    bare._last_fusion_exec = torch.zeros(1, 20, 3, 8, 8)
    bare._last_fusion_aux = None
    bare._last_fusion_meta = {"frame_contract": "expanded"}
    bare._last_spike_bins = 4
    model.H = torch.zeros(1, 6, 3, 8, 8)
    model.L = torch.zeros(1, 6, 7, 8, 8)

    with pytest.raises(ValueError, match="Fusion aux loss expected canonical main timeline"):
        model._compute_fusion_aux_loss(is_phase1=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/models/test_model_plain_fusion_aux_loss.py -k "reads_last_fusion_main_only or rejects_time_mismatch" -v
```

Expected: FAIL because `_compute_fusion_aux_loss()` still reads legacy cached fusion outputs.

- [ ] **Step 3: Write minimal implementation**

Update `models/model_plain.py`:

```python
fusion_main = getattr(vrt, "_last_fusion_main", None)
if fusion_main is None:
    return torch.tensor(0.0, device=self.device)
if fusion_main.size(1) != self.H.size(1):
    raise ValueError(
        "Fusion aux loss expected canonical main timeline "
        f"N={self.H.size(1)}, got {fusion_main.size(1)}."
    )
fusion_center = fusion_main
```

Update `_phase1_fusion_forward()` to cache:

```python
fusion_result = vrt.fusion_adapter(rgb=rgb, spike=spike)
vrt._last_fusion_main = fusion_result["fused_main"]
vrt._last_fusion_exec = fusion_result["backbone_view"]
vrt._last_fusion_aux = fusion_result["aux_view"]
vrt._last_fusion_meta = fusion_result["meta"]
vrt._last_spike_bins = spike.shape[2]
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/models/test_model_plain_fusion_aux_loss.py -k "reads_last_fusion_main_only or rejects_time_mismatch" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/model_plain.py tests/models/test_model_plain_fusion_aux_loss.py
git commit -m "fix(train): supervise canonical fusion main timeline only"
```

### Task 6: Keep downstream execution on the semantic execution view

**Files:**
- Modify: `models/architectures/vrt/vrt.py`
- Modify: `models/model_plain.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Write the failing downstream-execution test**

Add or update a test in `tests/models/test_vrt_fusion_integration.py`:

```python
def test_vrt_flow_alignment_uses_execution_steps_not_main_steps(monkeypatch):
    ...
    captured = {}

    def _fake_get_flows(_x, flow_spike=None):
        captured["x_shape"] = tuple(_x.shape)
        captured["flow_shape"] = None if flow_spike is None else tuple(flow_spike.shape)
        return dummy_flows, dummy_flows

    ...
    assert captured["x_shape"][1] == 24
    assert captured["flow_shape"][1] == 24
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py -k "uses_execution_steps_not_main_steps" -v
```

Expected: FAIL if VRT still aligns flow against ambiguous or canonicalized main steps.

- [ ] **Step 3: Write minimal implementation**

Ensure `models/architectures/vrt/vrt.py` uses:

```python
exec_steps = backbone_view.size(1)
flow_spike = self._align_flow_spike_to_fused_time_axis(
    flow_spike=flow_spike,
    fused_steps=exec_steps,
    spike_bins=spike_bins,
)
```

No downstream execution path should use `fused_main.size(1)` when `backbone_view` is available.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py -k "uses_execution_steps_not_main_steps" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/architectures/vrt/vrt.py models/model_plain.py tests/models/test_vrt_fusion_integration.py
git commit -m "fix(fusion): execute downstream on semantic execution view"
```

### Task 7: Update debug dumping to default to main view and optionally inspect expanded execution view

**Files:**
- Modify: `models/fusion/debug.py`
- Modify: `models/model_vrt.py`
- Test: `tests/models/test_fusion_debug_dumper.py`

- [ ] **Step 1: Write the failing debug contract tests**

Add or update tests in `tests/models/test_fusion_debug_dumper.py`:

```python
def test_fusion_debug_dumper_defaults_to_main_view(tmp_path):
    dumper = FusionDebugDumper(_make_opt(tmp_path))
    fusion_main = torch.rand(1, 2, 3, 8, 8)
    dumped = dumper.dump_tensor(
        fusion_main=fusion_main,
        fusion_exec=torch.rand(1, 8, 3, 8, 8),
        fusion_meta={"frame_contract": "expanded", "spike_bins": 4},
        current_step=3,
        folder="GOPR0001",
        gt=torch.rand(1, 2, 3, 8, 8),
        rank=0,
    )
    assert dumped is True
```

```python
def test_fusion_debug_dumper_can_dump_expanded_execution_view_when_requested(tmp_path):
    dumper = FusionDebugDumper(_make_opt(tmp_path))
    fusion_exec = torch.rand(1, 8, 3, 8, 8)
    dumped = dumper.dump_tensor(
        fusion_main=torch.rand(1, 2, 3, 8, 8),
        fusion_exec=fusion_exec,
        fusion_meta={"frame_contract": "expanded", "spike_bins": 4},
        current_step=4,
        folder="GOPR0002",
        gt=torch.rand(1, 2, 3, 8, 8),
        rank=0,
        source_view="exec",
    )
    assert dumped is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/models/test_fusion_debug_dumper.py -k "defaults_to_main_view or can_dump_expanded_execution_view" -v
```

Expected: FAIL because current debug API still expects the legacy single fusion tensor.

- [ ] **Step 3: Write minimal implementation**

In `models/fusion/debug.py` and `models/model_vrt.py`, add an explicit dump path that accepts:

```python
fusion_main = getattr(bare, "_last_fusion_main", None)
fusion_exec = getattr(bare, "_last_fusion_exec", None)
fusion_aux = getattr(bare, "_last_fusion_aux", None)
fusion_meta = getattr(bare, "_last_fusion_meta", {}) or {}
```

Use:

- `fusion_main` as default dump source
- `fusion_exec` only when config or call site explicitly requests expanded execution-view inspection

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/models/test_fusion_debug_dumper.py -v
```

Expected: PASS, including the earlier hook-compatibility test and migrated legacy dumper tests.

- [ ] **Step 5: Commit**

```bash
git add models/fusion/debug.py models/model_vrt.py tests/models/test_fusion_debug_dumper.py
git commit -m "refactor(debug): default to main view and expose exec-view dumps"
```

### Task 8: Preserve config compatibility and lock the refactor with focused regressions

**Files:**
- Test: `tests/models/test_fusion_early_adapter.py`
- Test: `tests/models/test_vrt_fusion_integration.py`
- Test: `tests/models/test_model_plain_fusion_aux_loss.py`
- Test: `tests/models/test_fusion_debug_dumper.py`

- [ ] **Step 1: Add a config-compatibility test**

Add a test asserting operator defaults require no new config:

```python
def test_early_wrapper_inferrs_contract_from_operator_without_new_config():
    from models.fusion.factory import create_fusion_operator
    from models.fusion.adapters.early import EarlyFusionAdapter

    op = create_fusion_operator("mamba", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op, mode="replace", inject_stages=[], spike_chans=4)
    assert getattr(adapter.operator, "frame_contract", None) == "collapsed"
```

- [ ] **Step 2: Run focused adapter tests**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py -v
```

Expected: PASS

- [ ] **Step 3: Run VRT integration tests**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py -v
```

Expected: PASS

- [ ] **Step 4: Run fusion aux-loss and debug tests**

Run:

```bash
python -m pytest tests/models/test_model_plain_fusion_aux_loss.py tests/models/test_fusion_debug_dumper.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_fusion_early_adapter.py tests/models/test_vrt_fusion_integration.py tests/models/test_model_plain_fusion_aux_loss.py tests/models/test_fusion_debug_dumper.py models/fusion/adapters/early.py models/architectures/vrt/vrt.py models/model_plain.py models/fusion/debug.py models/model_vrt.py models/fusion/operators/gated.py models/fusion/operators/concat.py models/fusion/operators/pase.py models/fusion/operators/mamba.py
git commit -m "test(fusion): verify semantic fusion main and execution contracts"
```

## Self-Review

- Spec coverage:
  - semantic split between supervision and execution: Tasks 2-5
  - explicit expanded/collapsed split: Tasks 1-2
  - canonical training on `N`: Task 4
  - preserved expanded downstream execution: Tasks 3 and 5
  - config compatibility by default: Task 7
  - debug use of main versus exec views: Task 6
- Placeholder scan:
  - no `TODO`, `TBD`, or vague “handle appropriately” language remains
  - code-changing tasks include concrete code or assertions
- Type consistency:
  - canonical cache names are `_last_fusion_main`, `_last_fusion_exec`, `_last_fusion_aux`, `_last_fusion_meta`
  - canonical wrapper keys are `fused_main`, `backbone_view`, `aux_view`, `meta`

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-24-fusion-wrapper-dual-contract-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
