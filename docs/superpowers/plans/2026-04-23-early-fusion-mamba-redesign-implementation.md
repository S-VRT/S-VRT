# Early Fusion Mamba Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current flattened early-fusion `mamba` path with an early-only, RGB-conditioned spike-sequence fusion operator that outputs one RGB-aligned fused frame per input frame and is validated end-to-end in VRT.

**Architecture:** Keep legacy early fusion behavior for `concat/gated/pase`, but fork `mamba` into a structured early-only path. The new `mamba` operator consumes `[B, T, 3, H, W]` RGB and `[B, T, S, H, W]` spike inputs, models spike-bin time inside fusion, produces `[B, T, 3, H, W]` outputs through residual correction, and rejects incompatible legacy full-T early semantics.

**Tech Stack:** Python, PyTorch, pytest, optional `mamba_ssm`, existing VRT/fusion factory code.

---

### Task 1: Lock in Early Mamba Operator Semantics with Failing Tests

**Files:**
- Modify: `tests/models/test_fusion_early_adapter.py`
- Test: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: Add failing tests for structured early `mamba` output shape and passthrough initialization**

```python
def test_mamba_operator_structured_early_shape_or_missing_dep():
    op = create_fusion_operator(
        "mamba",
        3,
        1,
        3,
        {
            "model_dim": 48,
            "d_state": 32,
            "d_conv": 4,
            "expand": 2,
            "num_layers": 3,
        },
    )
    rgb = torch.randn(2, 5, 3, 12, 12)
    spike = torch.randn(2, 5, 8, 12, 12)
    try:
        out = op(rgb, spike)
    except RuntimeError as exc:
        assert "mamba_ssm is required" in str(exc)
        return
    assert out.shape == (2, 5, 3, 12, 12)


def test_mamba_operator_passthrough_at_init_or_missing_dep():
    op = create_fusion_operator(
        "mamba",
        3,
        1,
        3,
        {
            "model_dim": 48,
            "d_state": 32,
            "d_conv": 4,
            "expand": 2,
            "num_layers": 3,
            "init_gate_bias": -5.0,
        },
    )
    rgb = torch.ones(1, 2, 3, 8, 8) * 0.5
    spike = torch.zeros(1, 2, 6, 8, 8)
    try:
        with torch.no_grad():
            out = op(rgb, spike)
    except RuntimeError as exc:
        assert "mamba_ssm is required" in str(exc)
        return
    assert torch.allclose(out, rgb, atol=1e-5)
```

- [ ] **Step 2: Add failing tests for structured early adapter behavior**

```python
class StructuredRecordingOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_rgb = None
        self.last_spike = None

    def forward(self, rgb, spike):
        self.last_rgb = rgb
        self.last_spike = spike
        return rgb


def test_early_adapter_mamba_keeps_frame_structure():
    op = StructuredRecordingOperator()
    setattr(op, "expects_structured_early", True)
    adapter = EarlyFusionAdapter(operator=op, spike_chans=8)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 12, 12)
    out = adapter(rgb=rgb, spike=spike)
    assert out.shape == (2, 6, 3, 12, 12)
    assert op.last_rgb.shape == (2, 6, 3, 12, 12)
    assert op.last_spike.shape == (2, 6, 8, 12, 12)


def test_early_adapter_mamba_upsamples_without_flattening():
    op = StructuredRecordingOperator()
    setattr(op, "expects_structured_early", True)
    adapter = EarlyFusionAdapter(operator=op, spike_chans=8)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 6, 6)
    out = adapter(rgb=rgb, spike=spike)
    assert out.shape == (2, 6, 3, 12, 12)
    assert op.last_spike.shape == (2, 6, 8, 12, 12)
```

- [ ] **Step 3: Run the targeted tests to confirm they fail first**

Run: `pytest tests/models/test_fusion_early_adapter.py::test_mamba_operator_structured_early_shape_or_missing_dep tests/models/test_fusion_early_adapter.py::test_mamba_operator_passthrough_at_init_or_missing_dep tests/models/test_fusion_early_adapter.py::test_early_adapter_mamba_keeps_frame_structure tests/models/test_fusion_early_adapter.py::test_early_adapter_mamba_upsamples_without_flattening -v`
Expected: FAIL because the current `mamba` operator expects flattened semantics and the early adapter always expands `T*S`.

- [ ] **Step 4: Commit the failing tests**

```bash
git add tests/models/test_fusion_early_adapter.py
git commit -m "test(fusion): lock structured early mamba semantics"
```


### Task 2: Rebuild `MambaFusionOperator` as RGB-Conditioned Residual Early Fusion

**Files:**
- Modify: `models/fusion/operators/mamba.py`
- Test: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: Replace the current operator with a structured residual design**

```python
from typing import Dict

import torch
from torch import nn


class _MambaBlock(nn.Module):
    def __init__(self, model_dim: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        try:
            from mamba_ssm import Mamba  # type: ignore
        except (ImportError, ModuleNotFoundError):
            self.mamba = None
            return
        self.norm = nn.LayerNorm(model_dim)
        self.mamba = Mamba(d_model=model_dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.mamba is None:
            raise RuntimeError("mamba_ssm is required for mamba fusion operator.")
        return tokens + self.mamba(self.norm(tokens))


class MambaFusionOperator(nn.Module):
    expects_structured_early = True

    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int, operator_params: Dict):
        super().__init__()
        if rgb_chans != 3:
            raise ValueError("MambaFusionOperator requires rgb_chans=3.")
        if spike_chans != 1:
            raise ValueError("MambaFusionOperator requires spike_chans=1 at construction time.")
        if out_chans != 3:
            raise ValueError("MambaFusionOperator requires out_chans=3.")

        self.rgb_chans = rgb_chans
        self.spike_chans = spike_chans
        self.out_chans = out_chans
        self.operator_params = operator_params

        model_dim = int(operator_params.get("model_dim", 48))
        d_state = int(operator_params.get("d_state", 32))
        d_conv = int(operator_params.get("d_conv", 4))
        expand = int(operator_params.get("expand", 2))
        num_layers = int(operator_params.get("num_layers", 3))
        init_gate_bias = float(operator_params.get("init_gate_bias", -5.0))

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, model_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1),
        )
        self.spike_token_proj = nn.Conv2d(1, model_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [_MambaBlock(model_dim=model_dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)]
        )
        self.correction_head = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_dim, 3, kernel_size=1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_dim, 3, kernel_size=1),
        )

        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, init_gate_bias)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() != 5:
            raise ValueError("mamba early fusion expects rgb with shape [B, T, 3, H, W].")
        if spike_feat.dim() != 5:
            raise ValueError("mamba early fusion expects spike with shape [B, T, S, H, W].")

        bsz, steps, rgb_chans, height, width = rgb_feat.shape
        spike_bsz, spike_steps, spike_bins, spike_height, spike_width = spike_feat.shape
        if (bsz, steps, height, width) != (spike_bsz, spike_steps, spike_height, spike_width):
            raise ValueError("rgb and spike must share batch, time, height, and width dimensions")
        if rgb_chans != 3:
            raise ValueError(f"Expected rgb channels=3, got {rgb_chans}")

        rgb_flat = rgb_feat.reshape(bsz * steps, 3, height, width)
        rgb_ctx = self.rgb_encoder(rgb_flat).reshape(bsz, steps, -1, height, width)

        spike_flat = spike_feat.reshape(bsz * steps * spike_bins, 1, height, width)
        spike_tokens = self.spike_token_proj(spike_flat).reshape(bsz, steps, spike_bins, -1, height, width)
        tokens = spike_tokens + rgb_ctx.unsqueeze(2)

        model_dim = tokens.size(3)
        seq = tokens.permute(0, 1, 4, 5, 2, 3).reshape(bsz * steps * height * width, spike_bins, model_dim)
        for block in self.blocks:
            seq = block(seq)
        pooled = seq.mean(dim=1).reshape(bsz, steps, height, width, model_dim).permute(0, 1, 4, 2, 3)

        pooled_flat = pooled.reshape(bsz * steps, model_dim, height, width)
        correction = self.correction_head(pooled_flat).reshape(bsz, steps, 3, height, width)
        gate = torch.sigmoid(self.gate_head(pooled_flat)).reshape(bsz, steps, 3, height, width)
        return rgb_feat + gate * correction
```

- [ ] **Step 2: Run the focused early adapter tests**

Run: `pytest tests/models/test_fusion_early_adapter.py::test_mamba_operator_structured_early_shape_or_missing_dep tests/models/test_fusion_early_adapter.py::test_mamba_operator_passthrough_at_init_or_missing_dep -v`
Expected: PASS, or PASS for missing-dependency guard when `mamba_ssm` is unavailable.

- [ ] **Step 3: Commit the operator redesign**

```bash
git add models/fusion/operators/mamba.py tests/models/test_fusion_early_adapter.py
git commit -m "feat(fusion): redesign early mamba as rgb-conditioned residual operator"
```


### Task 3: Fork the Early Adapter for Structured `mamba` Input

**Files:**
- Modify: `models/fusion/adapters/early.py`
- Test: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: Add a structured path keyed off the operator capability flag**

```python
class EarlyFusionAdapter(nn.Module):
    def __init__(self, operator: nn.Module, mode: str = "replace", inject_stages: Optional[list] = None, spike_chans: Optional[int] = None, **kwargs: Any):
        super().__init__()
        self.operator = operator
        self.mode = mode
        self.inject_stages = inject_stages if inject_stages is not None else []
        self.spike_chans = spike_chans
        self.spike_upsample = SpikeUpsample(spike_chans) if spike_chans is not None else None
        self.expects_structured_early = bool(getattr(operator, "expects_structured_early", False))
        self.kwargs = kwargs

    def forward(self, rgb: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        if rgb.dim() != 5:
            raise ValueError("rgb must be 5D tensor [B, N, C, H, W]")
        if spike.dim() != 5:
            raise ValueError("spike must be 5D tensor [B, N, S, H, W]")

        bsz, steps, _, height, width = rgb.shape
        spike_bsz, spike_steps, spike_steps_per_frame, spike_height, spike_width = spike.shape
        if (bsz, steps) != (spike_bsz, spike_steps):
            raise ValueError("rgb and spike must share batch size and steps")

        if (spike_height, spike_width) != (height, width):
            if self.spike_upsample is None:
                raise ValueError("Cannot upsample spike features to match rgb spatial dimensions without spike_chans.")
            spike_flat = spike.reshape(bsz * steps, spike_steps_per_frame, spike_height, spike_width)
            spike_flat = self.spike_upsample(spike_flat, target_h=height, target_w=width)
            spike = spike_flat.reshape(bsz, steps, spike_steps_per_frame, height, width)

        if self.expects_structured_early:
            return self.operator(rgb, spike)

        rgb_rep = rgb.unsqueeze(2).expand(bsz, steps, spike_steps_per_frame, rgb.shape[2], height, width)
        rgb_rep = rgb_rep.reshape(bsz, steps * spike_steps_per_frame, rgb.shape[2], height, width)
        spk = spike.reshape(bsz, steps * spike_steps_per_frame, 1, height, width)
        return self.operator(rgb_rep, spk)
```

- [ ] **Step 2: Run adapter tests to verify legacy and structured paths coexist**

Run: `pytest tests/models/test_fusion_early_adapter.py -v`
Expected: PASS, except any tests that still assume old `mamba` flattening semantics and therefore need to be updated in-place.

- [ ] **Step 3: Commit the adapter fork**

```bash
git add models/fusion/adapters/early.py tests/models/test_fusion_early_adapter.py
git commit -m "feat(fusion): fork early adapter for structured mamba path"
```


### Task 4: Enforce VRT Early-Only `mamba` Constraints with Tests

**Files:**
- Modify: `tests/models/test_vrt_fusion_integration.py`
- Modify: `models/architectures/vrt/vrt.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Add failing VRT integration tests for valid and invalid `early+mamba` configs**

```python
def test_vrt_builds_with_structured_early_mamba_config():
    model = VRT(
        upscale=1,
        img_size=[2, 8, 8],
        window_size=[2, 8, 8],
        depths=[1] * 7,
        indep_reconsts=[11, 12],
        embed_dims=[8] * 8,
        num_heads=[1] * 8,
        pa_frames=0,
        deformable_groups=1,
        nonblind_denoising=False,
        in_chans=11,
        input_mode="dual",
        optical_flow_config={"type": "spynet"},
        opt={
            "netG": {
                "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 11},
                "fusion": {
                    "placement": "early",
                    "operator": "mamba",
                    "out_chans": 3,
                    "operator_params": {"model_dim": 48, "d_state": 32, "d_conv": 4, "expand": 2, "num_layers": 3},
                    "early": {"expand_to_full_t": False},
                },
            }
        },
    )
    assert model.fusion_operator is not None
    assert getattr(model.fusion_operator, "expects_structured_early", False) is True


def test_vrt_rejects_mamba_with_full_t_early_expansion():
    with pytest.raises(ValueError, match="mamba.*expand_to_full_t"):
        VRT(
            upscale=1,
            img_size=[2, 8, 8],
            window_size=[2, 8, 8],
            depths=[1] * 7,
            indep_reconsts=[11, 12],
            embed_dims=[8] * 8,
            num_heads=[1] * 8,
            pa_frames=0,
            deformable_groups=1,
            nonblind_denoising=False,
            in_chans=11,
            input_mode="dual",
            optical_flow_config={"type": "spynet"},
            opt={
                "netG": {
                    "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 11},
                    "fusion": {
                        "placement": "early",
                        "operator": "mamba",
                        "out_chans": 3,
                        "early": {"expand_to_full_t": True},
                    },
                }
            },
        )
```

- [ ] **Step 2: Implement explicit VRT validation for `mamba`**

```python
operator_name = str(fusion_cfg.get('operator', 'concat')).strip().lower()
operator_params = fusion_cfg.get('operator_params', {})

if operator_name == 'mamba':
    if fusion_placement != 'early':
        raise ValueError("fusion.operator='mamba' requires fusion.placement='early'.")
    if early_out_chans != 3:
        raise ValueError("fusion.operator='mamba' requires fusion.out_chans=3 for early fusion.")
    if bool(early_cfg.get('expand_to_full_t', False)):
        raise ValueError("fusion.operator='mamba' does not support fusion.early.expand_to_full_t=true.")
```

Keep the existing `effective_in_chans = early_out_chans` behavior so the VRT backbone still sees frame-level 3-channel fused output.

- [ ] **Step 3: Run VRT integration tests**

Run: `pytest tests/models/test_vrt_fusion_integration.py::test_vrt_builds_with_structured_early_mamba_config tests/models/test_vrt_fusion_integration.py::test_vrt_rejects_mamba_with_full_t_early_expansion -v`
Expected: PASS

- [ ] **Step 4: Commit the VRT validation work**

```bash
git add models/architectures/vrt/vrt.py tests/models/test_vrt_fusion_integration.py
git commit -m "feat(vrt): validate structured early mamba fusion config"
```


### Task 5: Add Factory/Config Regression Coverage and Final Verification

**Files:**
- Modify: `tests/models/test_fusion_factory.py`
- Modify: `tests/models/test_fusion_early_adapter.py`
- Modify: `tests/models/test_vrt_fusion_integration.py`
- Test: `tests/models/test_fusion_factory.py`
- Test: `tests/models/test_fusion_early_adapter.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Add focused regression tests for `mamba` construction constraints**

```python
def test_mamba_operator_rejects_non_rgb3():
    with pytest.raises(ValueError, match="rgb_chans=3"):
        create_fusion_operator("mamba", 8, 1, 3, {})


def test_mamba_operator_rejects_non_out3():
    with pytest.raises(ValueError, match="out_chans=3"):
        create_fusion_operator("mamba", 3, 1, 8, {})
```

- [ ] **Step 2: Run the full fusion regression suite**

Run: `pytest tests/models/test_fusion_factory.py tests/models/test_fusion_early_adapter.py tests/models/test_vrt_fusion_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit the regression coverage**

```bash
git add tests/models/test_fusion_factory.py tests/models/test_fusion_early_adapter.py tests/models/test_vrt_fusion_integration.py
git commit -m "test(fusion): cover structured early mamba constraints"
```

- [ ] **Step 4: Run final verification**

Run: `pytest tests/models/test_fusion_factory.py tests/models/test_fusion_early_adapter.py tests/models/test_vrt_fusion_integration.py -q`
Expected: all selected tests PASS

- [ ] **Step 5: Summarize verification evidence in the handoff**

```text
Record the exact pytest command used, whether `mamba_ssm` was available, and which tests exercised fallback behavior versus structured execution.
```


## Self-Review

### 1. Spec coverage

1. Structured early-only `mamba` semantics: Tasks 1, 2, 3.
2. RGB-conditioned residual design with conservative initialization: Task 2.
3. Early adapter structured path only for `mamba`: Task 3.
4. VRT config constraints (`early` only, `out_chans=3`, no `expand_to_full_t`): Task 4.
5. Regression and dependency/error handling coverage: Tasks 1, 5.

No uncovered spec requirement found.

### 2. Placeholder scan

Searched for: `TBD`, `TODO`, `implement later`, `add validation`, `similar to`.
None used as unresolved placeholders.

### 3. Type consistency

1. `mamba` operator contract is consistently defined as `rgb: [B, T, 3, H, W]`, `spike: [B, T, S, H, W]`, output `[B, T, 3, H, W]`.
2. Structured early detection uses one consistent capability flag: `expects_structured_early`.
3. VRT config constraints consistently refer to `fusion.operator='mamba'`, `fusion.placement='early'`, `fusion.out_chans=3`, and `fusion.early.expand_to_full_t`.
