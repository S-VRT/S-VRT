# Configurable Mamba Fusion Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-controlled early-fusion frame contract so `mamba` can run either collapsed or expanded for ablation.

**Architecture:** `EarlyFusionAdapter` receives an optional `frame_contract` override from `fusion.early.frame_contract`. The adapter resolves `operator_default`, `collapsed`, or `expanded`, packages tensors accordingly, and records requested/effective contracts in metadata. VRT validates the new field and stops rejecting `mamba` expanded ablations.

**Tech Stack:** PyTorch, pytest, existing S-VRT fusion adapter/operator architecture.

---

## File Structure

- Modify `models/fusion/adapters/early.py`: parse and resolve the frame-contract override, validate collapsed support, and emit metadata.
- Modify `models/fusion/adapters/__init__.py`: pass `frame_contract` from adapter factory to `EarlyFusionAdapter`.
- Modify `models/architectures/vrt/vrt.py`: read `fusion.early.frame_contract`, validate mamba constraints using effective contract, and pass override through factory.
- Modify `tests/models/test_fusion_early_adapter.py`: add adapter-level tests for default, override, invalid values, and metadata.
- Modify `tests/models/test_vrt_fusion_integration.py`: add VRT construction test for mamba expanded config.
- Modify `options/gopro_rgbspike_server.json`: add explicit default `"frame_contract": "operator_default"` near existing mamba fusion config.

### Task 1: Adapter Contract Override

**Files:**
- Modify: `tests/models/test_fusion_early_adapter.py`
- Modify: `models/fusion/adapters/early.py`

- [ ] **Step 1: Write failing adapter tests**

Add tests that assert:

```python
def test_early_adapter_frame_contract_override_expands_collapsed_operator():
    op = StructuredRecordingOperator()
    adapter = EarlyFusionAdapter(
        operator=op,
        spike_chans=4,
        frame_contract="expanded",
    )
    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)

    result = adapter(rgb=rgb, spike=spike)

    assert op.last_rgb.shape == (1, 8, 3, 8, 8)
    assert op.last_spike.shape == (1, 8, 1, 8, 8)
    assert result["fused_main"].shape == (1, 2, 3, 8, 8)
    assert result["backbone_view"].shape == (1, 8, 3, 8, 8)
    assert result["meta"]["requested_frame_contract"] == "expanded"
    assert result["meta"]["frame_contract"] == "expanded"
```

```python
def test_early_adapter_default_contract_uses_operator_contract():
    op = StructuredRecordingOperator()
    adapter = EarlyFusionAdapter(operator=op, spike_chans=4)
    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)

    result = adapter(rgb=rgb, spike=spike)

    assert op.last_rgb.shape == (1, 2, 3, 8, 8)
    assert op.last_spike.shape == (1, 2, 4, 8, 8)
    assert result["meta"]["requested_frame_contract"] == "operator_default"
    assert result["meta"]["frame_contract"] == "collapsed"
```

```python
def test_early_adapter_rejects_invalid_frame_contract():
    with pytest.raises(ValueError, match="frame_contract"):
        EarlyFusionAdapter(operator=RecordingOperator(), frame_contract="bad")
```

```python
def test_early_adapter_collapsed_override_requires_structured_support():
    with pytest.raises(ValueError, match="collapsed"):
        EarlyFusionAdapter(operator=RecordingOperator(), frame_contract="collapsed")
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
uv run pytest tests/models/test_fusion_early_adapter.py::test_early_adapter_frame_contract_override_expands_collapsed_operator tests/models/test_fusion_early_adapter.py::test_early_adapter_default_contract_uses_operator_contract tests/models/test_fusion_early_adapter.py::test_early_adapter_rejects_invalid_frame_contract tests/models/test_fusion_early_adapter.py::test_early_adapter_collapsed_override_requires_structured_support -q
```

Expected: fail because `EarlyFusionAdapter` does not accept `frame_contract`.

- [ ] **Step 3: Implement adapter support**

In `EarlyFusionAdapter.__init__`, add `frame_contract: str = "operator_default"` and validate:

```python
self.operator_frame_contract = str(getattr(operator, "frame_contract", "expanded")).strip().lower()
self.requested_frame_contract = str(frame_contract or "operator_default").strip().lower()
allowed_contracts = {"operator_default", "collapsed", "expanded"}
if self.requested_frame_contract not in allowed_contracts:
    raise ValueError(
        f"Unsupported frame_contract={frame_contract!r}; "
        "expected one of: operator_default, collapsed, expanded"
    )
self.frame_contract = (
    self.operator_frame_contract
    if self.requested_frame_contract == "operator_default"
    else self.requested_frame_contract
)
self.expects_structured_early = bool(
    getattr(operator, "expects_structured_early", False)
) or self.operator_frame_contract == "collapsed"
if self.frame_contract == "collapsed" and not self.expects_structured_early:
    raise ValueError("frame_contract='collapsed' requires an operator that supports structured early fusion.")
```

Update `_build_meta()` to include `requested_frame_contract`.

In `forward()`, use `self.frame_contract` as the effective contract instead of reading the operator every time.

- [ ] **Step 4: Run adapter tests and verify green**

Run the same pytest command from Step 2.

Expected: pass.

### Task 2: Factory and VRT Config Wiring

**Files:**
- Modify: `models/fusion/adapters/__init__.py`
- Modify: `models/architectures/vrt/vrt.py`
- Modify: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Write failing VRT config test**

Add:

```python
def test_vrt_allows_mamba_expanded_frame_contract_config():
    opt = {
        "netG": {
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 7,
            },
            "output_mode": "restoration",
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "mamba",
                "out_chans": 3,
                "early": {
                    "frame_contract": "expanded",
                    "expand_to_full_t": True,
                },
                "operator_params": {},
            },
        }
    }

    model = VRT(
        upscale=1,
        in_chans=7,
        out_chans=3,
        img_size=[2, 8, 8],
        window_size=[2, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=opt,
    )

    assert model.fusion_adapter.requested_frame_contract == "expanded"
    assert model.fusion_adapter.frame_contract == "expanded"
```

- [ ] **Step 2: Run test and verify red**

Run:

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py::test_vrt_allows_mamba_expanded_frame_contract_config -q
```

Expected: fail because VRT still rejects `mamba + expand_to_full_t=true` or does not pass the frame contract to the adapter.

- [ ] **Step 3: Wire config through factory**

In `create_fusion_adapter()`, accept `frame_contract` in `**kwargs` and pass it to `EarlyFusionAdapter`.

In `VRT.__init__`:

```python
requested_frame_contract = str(early_cfg.get("frame_contract", "operator_default")).strip().lower()
if requested_frame_contract not in {"operator_default", "collapsed", "expanded"}:
    raise ValueError(...)
operator_default_contract = "collapsed" if operator_name == "mamba" else "expanded"
effective_frame_contract = (
    operator_default_contract
    if requested_frame_contract == "operator_default"
    else requested_frame_contract
)
```

Only reject `mamba + expand_to_full_t=true` when `effective_frame_contract == "collapsed"`.

Pass `frame_contract=requested_frame_contract` into early and hybrid adapter factory calls.

- [ ] **Step 4: Run VRT config test and verify green**

Run the command from Step 2.

Expected: pass.

### Task 3: Config Default and Regression Tests

**Files:**
- Modify: `options/gopro_rgbspike_server.json`
- Test: `tests/models/test_fusion_early_adapter.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Add explicit default config**

Under the active mamba fusion early config, add:

```json
"frame_contract": "operator_default"
```

- [ ] **Step 2: Run focused regression tests**

Run:

```bash
uv run pytest tests/models/test_fusion_early_adapter.py tests/models/test_vrt_fusion_integration.py -q
```

Expected: pass.

- [ ] **Step 3: Run syntax check**

Run:

```bash
python -m py_compile models/fusion/adapters/early.py models/fusion/adapters/__init__.py models/architectures/vrt/vrt.py
```

Expected: no output and exit 0.

## Self-Review

- Spec coverage: tasks cover config field, default behavior, expanded mamba ablation, validation, metadata, VRT integration, and config default.
- Placeholder scan: no placeholder steps remain.
- Type consistency: `frame_contract`, `requested_frame_contract`, and effective `frame_contract` are used consistently.
