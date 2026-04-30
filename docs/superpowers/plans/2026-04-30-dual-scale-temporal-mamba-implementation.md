# Dual-Scale Temporal Mamba Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new early-fusion operator, `dual_scale_temporal_mamba`, that consumes per-frame raw spike windows, models intra-frame raw timing and inter-frame evolution with two Mamba stages, and remains trainable within the project's current budget.

**Architecture:** Reuse the existing structured early-fusion contract and raw-window dataset path. Add a new fusion operator that first patchifies raw spike windows and runs a local Mamba over the raw temporal axis `L`, then compresses each frame into summary tokens and runs a second lightweight Mamba over the frame axis `T`, before writing back a conservative RGB residual. Wire VRT so raw-window input is legal for both `pase_residual` and the new operator, surface metadata for analysis, and add one runnable config for the first budget-controlled experiment.

**Tech Stack:** Python, PyTorch, NumPy, pytest, existing VRT/fusion adapter stack, existing raw-window spike representation pipeline

---

## File Map

- Create: `models/fusion/operators/dual_scale_temporal_mamba.py`
  - New hierarchical local-to-global Mamba fusion operator with raw-window spike input, diagnostics, and residual write-back.
- Modify: `models/fusion/operators/__init__.py`
  - Register `dual_scale_temporal_mamba` in the operator factory and exports.
- Create: `tests/models/test_dual_scale_temporal_mamba.py`
  - Focused operator-level tests using a fake Mamba block to avoid CUDA/Mamba dependency in unit coverage.
- Modify: `models/architectures/vrt/vrt.py`
  - Allow raw-window spike input for the new operator, pass full spike channels into the operator constructor, and record operator-specific metadata.
- Modify: `tests/models/test_vrt_fusion_integration.py`
  - VRT constructor and forward integration tests for `dual_scale_temporal_mamba + raw_window`.
- Create: `options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json`
  - Runnable training config built from the existing `pase_residual` raw-window config with the new operator and budget-constrained params.
- Modify: `tests/e2e/test_gopro_rgbspike_server_e2e.py`
  - Config-level guard that parses the new option file and checks the intended operator/raw-window wiring.

## Task 1: Define Operator Contract with Focused Unit Tests

**Files:**
- Create: `tests/models/test_dual_scale_temporal_mamba.py`

- [ ] **Step 1: Write the failing operator tests**

Create `tests/models/test_dual_scale_temporal_mamba.py` with the following tests:

```python
import pytest
import torch

from models.fusion.factory import create_fusion_operator


def test_factory_builds_dual_scale_temporal_mamba():
    operator = create_fusion_operator(
        "dual_scale_temporal_mamba",
        rgb_chans=3,
        spike_chans=21,
        out_chans=3,
        operator_params={"token_dim": 8, "patch_stride": 4, "local_layers": 1, "global_layers": 1},
    )

    assert operator.__class__.__name__ == "DualScaleTemporalMambaFusionOperator"
    assert operator.frame_contract == "collapsed"
    assert operator.expects_structured_early is True
    assert operator.spike_chans == 21


def test_dual_scale_temporal_mamba_rejects_non_rgb3():
    with pytest.raises(ValueError, match="rgb_chans=3"):
        create_fusion_operator(
            "dual_scale_temporal_mamba",
            rgb_chans=4,
            spike_chans=21,
            out_chans=3,
            operator_params={},
        )


def test_dual_scale_temporal_mamba_rejects_non_positive_spike_chans():
    with pytest.raises(ValueError, match="spike_chans>0"):
        create_fusion_operator(
            "dual_scale_temporal_mamba",
            rgb_chans=3,
            spike_chans=0,
            out_chans=3,
            operator_params={},
        )


def test_dual_scale_temporal_mamba_shape_contract_or_missing_dep():
    operator = create_fusion_operator(
        "dual_scale_temporal_mamba",
        rgb_chans=3,
        spike_chans=21,
        out_chans=3,
        operator_params={"token_dim": 8, "patch_stride": 4, "local_layers": 1, "global_layers": 1},
    )
    rgb = torch.randn(1, 3, 3, 16, 16)
    spike = torch.randn(1, 3, 21, 16, 16)

    try:
        out = operator(rgb, spike)
    except RuntimeError as exc:
        assert "mamba_ssm is required" in str(exc)
    else:
        assert out.shape == rgb.shape


def test_dual_scale_temporal_mamba_exposes_diagnostics_or_missing_dep():
    operator = create_fusion_operator(
        "dual_scale_temporal_mamba",
        rgb_chans=3,
        spike_chans=21,
        out_chans=3,
        operator_params={
            "token_dim": 8,
            "patch_stride": 4,
            "local_layers": 1,
            "global_layers": 1,
            "enable_diagnostics": True,
        },
    )
    rgb = torch.randn(1, 2, 3, 16, 16)
    spike = torch.randn(1, 2, 21, 16, 16)

    try:
        _ = operator(rgb, spike)
    except RuntimeError as exc:
        assert "mamba_ssm is required" in str(exc)
    else:
        diagnostics = operator.diagnostics()
        assert "local_norm" in diagnostics
        assert "global_norm" in diagnostics
        assert "summary_gate_mean" in diagnostics
        assert "effective_update_norm" in diagnostics
```

- [ ] **Step 2: Run the focused operator tests and confirm they fail**

Run:

```bash
python -m pytest tests/models/test_dual_scale_temporal_mamba.py -v
```

Expected: FAIL with `Unknown fusion operator: dual_scale_temporal_mamba`.

- [ ] **Step 3: Commit the failing test file**

```bash
git add tests/models/test_dual_scale_temporal_mamba.py
git commit -m "test(fusion): define dual-scale temporal mamba contract"
```

## Task 2: Implement and Register the New Dual-Scale Operator

**Files:**
- Create: `models/fusion/operators/dual_scale_temporal_mamba.py`
- Modify: `models/fusion/operators/__init__.py`

- [ ] **Step 1: Implement the operator module**

Create `models/fusion/operators/dual_scale_temporal_mamba.py` with this implementation:

```python
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from contextlib import contextmanager, nullcontext


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
            raise RuntimeError("mamba_ssm is required for dual_scale_temporal_mamba fusion operator.")
        if not tokens.is_cuda:
            raise RuntimeError("mamba_ssm is required for dual_scale_temporal_mamba fusion operator with CUDA tensors.")
        return tokens + self.mamba(self.norm(tokens))


class _ResidualMambaStage(nn.Module):
    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([_MambaBlock(dim, d_state, d_conv, expand) for _ in range(depth)])
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            seq = block(seq)
        return seq + self.ffn(self.ffn_norm(seq))


class DualScaleTemporalMambaFusionOperator(nn.Module):
    expects_structured_early = True
    frame_contract = "collapsed"

    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int, operator_params: Dict):
        super().__init__()
        if rgb_chans != 3:
            raise ValueError("DualScaleTemporalMambaFusionOperator requires rgb_chans=3.")
        if spike_chans <= 0:
            raise ValueError("DualScaleTemporalMambaFusionOperator requires spike_chans>0.")
        if out_chans != 3:
            raise ValueError("DualScaleTemporalMambaFusionOperator requires out_chans=3.")

        self.spike_chans = spike_chans
        token_dim = int(operator_params.get("token_dim", 48))
        patch_stride = int(operator_params.get("patch_stride", 4))
        d_state = int(operator_params.get("d_state", 32))
        d_conv = int(operator_params.get("d_conv", 4))
        expand = int(operator_params.get("expand", 2))
        local_layers = int(operator_params.get("local_layers", 1))
        global_layers = int(operator_params.get("global_layers", 1))
        alpha_init = float(operator_params.get("alpha_init", 0.05))
        gate_bias_init = float(operator_params.get("gate_bias_init", -2.0))
        self.enable_diagnostics = bool(operator_params.get("enable_diagnostics", False))

        self.spike_projector = nn.Sequential(
            nn.Conv2d(1, token_dim, kernel_size=3, stride=patch_stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
        )
        self.rgb_context_encoder = nn.Sequential(
            nn.Conv2d(3, token_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.local_stage = _ResidualMambaStage(token_dim, d_state, d_conv, expand, local_layers)
        self.summary_gate = nn.Linear(token_dim, 1)
        self.global_stage = _ResidualMambaStage(token_dim, d_state, d_conv, expand, global_layers)
        self.fusion_body = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Conv2d(token_dim, 3, kernel_size=1)
        self.gate_head = nn.Conv2d(token_dim, 3, kernel_size=1)
        self.alpha = nn.Parameter(torch.full((1, 3, 1, 1), alpha_init))
        self._last_diagnostics = {"warmup_stage": "full"}
        self.timer = None

        nn.init.normal_(self.delta_head.weight, std=1e-3)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.normal_(self.gate_head.weight, std=1e-3)
        nn.init.constant_(self.gate_head.bias, gate_bias_init)

    def diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    def set_timer(self, timer) -> None:
        self.timer = timer

    def _timer(self, name: str):
        if self.timer is None:
            return nullcontext()
        return self.timer.timer(name)

    @contextmanager
    def _profiled_timer(self, name: str):
        if self.timer is None:
            yield
            return
        range_ctx = self.timer.profile_range(name) if hasattr(self.timer, "profile_range") else nullcontext()
        with range_ctx:
            with self._timer(name):
                yield

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() != 5:
            raise ValueError("dual_scale_temporal_mamba expects rgb with shape [B,T,3,H,W].")
        if spike_feat.dim() != 5:
            raise ValueError("dual_scale_temporal_mamba expects spike with shape [B,T,S,H,W].")

        bsz, steps, rgb_chans, height, width = rgb_feat.shape
        spike_bsz, spike_steps, spike_chans, spike_h, spike_w = spike_feat.shape
        if (bsz, steps, height, width) != (spike_bsz, spike_steps, spike_h, spike_w):
            raise ValueError("rgb and spike must share batch, time, height, and width dimensions")
        if rgb_chans != 3:
            raise ValueError(f"Expected rgb channels=3, got {rgb_chans}")
        if spike_chans != self.spike_chans:
            raise ValueError(f"Expected spike channels={self.spike_chans}, got {spike_chans}")

        with self._profiled_timer("dual_scale_spike_project"):
            spike_flat = spike_feat.reshape(bsz * steps * spike_chans, 1, height, width)
            spike_low = self.spike_projector(spike_flat)
            _, token_dim, low_h, low_w = spike_low.shape
            patch_tokens = low_h * low_w
            spike_low = spike_low.reshape(bsz, steps, spike_chans, token_dim, low_h, low_w)

        with self._profiled_timer("dual_scale_local_mamba"):
            local_seq = spike_low.permute(0, 1, 4, 5, 2, 3).reshape(bsz * steps * patch_tokens, spike_chans, token_dim)
            local_seq = self.local_stage(local_seq)

        with self._profiled_timer("dual_scale_summary"):
            gate_logits = self.summary_gate(local_seq)
            gate = torch.softmax(gate_logits, dim=1)
            frame_summary = (gate * local_seq).sum(dim=1).reshape(bsz, steps, patch_tokens, token_dim)

        with self._profiled_timer("dual_scale_global_mamba"):
            global_seq = frame_summary.permute(0, 2, 1, 3).reshape(bsz * patch_tokens, steps, token_dim)
            global_seq = self.global_stage(global_seq)
            global_feat = global_seq.reshape(bsz, patch_tokens, steps, token_dim).permute(0, 2, 3, 1)
            spike_ctx = global_feat.reshape(bsz * steps, token_dim, low_h, low_w)
            spike_ctx = F.interpolate(spike_ctx, size=(height, width), mode="bilinear", align_corners=False)

        with self._profiled_timer("dual_scale_writeback"):
            rgb_ctx = self.rgb_context_encoder(rgb_feat.reshape(bsz * steps, 3, height, width))
            fused = self.fusion_body(rgb_ctx + spike_ctx)
            delta = self.delta_head(fused).reshape(bsz, steps, 3, height, width)
            gate_logits = self.gate_head(fused).reshape(bsz, steps, 3, height, width)
            gate_map = torch.sigmoid(gate_logits)
            effective_update = self.alpha.view(1, 1, 3, 1, 1) * gate_map * delta
            out = rgb_feat + effective_update

        if self.enable_diagnostics:
            self._last_diagnostics = {
                "local_norm": float(local_seq.detach().float().norm(dim=-1).mean().item()),
                "global_norm": float(global_seq.detach().float().norm(dim=-1).mean().item()),
                "summary_gate_mean": float(gate.detach().float().mean().item()),
                "effective_update_norm": float(effective_update.detach().float().abs().mean().item()),
                "warmup_stage": "full",
            }
        else:
            self._last_diagnostics = {"warmup_stage": "full"}
        return out


__all__ = ["DualScaleTemporalMambaFusionOperator"]
```

- [ ] **Step 2: Register the operator in the factory**

Update `models/fusion/operators/__init__.py` with these edits:

```python
from .dual_scale_temporal_mamba import DualScaleTemporalMambaFusionOperator
```

```python
    if normalized_name == 'dual_scale_temporal_mamba':
        return DualScaleTemporalMambaFusionOperator(
            rgb_chans=rgb_chans,
            spike_chans=spike_chans,
            out_chans=out_chans,
            operator_params=operator_params,
        )
```

```python
    'DualScaleTemporalMambaFusionOperator',
```

- [ ] **Step 3: Re-run the focused operator tests and confirm they pass**

Run:

```bash
python -m pytest tests/models/test_dual_scale_temporal_mamba.py -v
```

Expected: PASS on constructor/factory tests, and either PASS on forward tests or PASS via the existing `"mamba_ssm is required"` guard branch.

- [ ] **Step 4: Commit**

```bash
git add models/fusion/operators/dual_scale_temporal_mamba.py models/fusion/operators/__init__.py tests/models/test_dual_scale_temporal_mamba.py
git commit -m "feat(fusion): add dual-scale temporal mamba operator"
```

## Task 3: Wire VRT Construction, Legality Checks, and Metadata

**Files:**
- Modify: `models/architectures/vrt/vrt.py`
- Modify: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Write the failing VRT integration tests**

Append these tests to `tests/models/test_vrt_fusion_integration.py`:

```python
def _dual_scale_raw_window_opt(in_chans=21 + 3, raw_window_length=21):
    return {
        "netG": {
            "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": in_chans},
            "fusion": {
                "placement": "early",
                "operator": "dual_scale_temporal_mamba",
                "out_chans": 3,
                "operator_params": {
                    "token_dim": 8,
                    "patch_stride": 4,
                    "local_layers": 1,
                    "global_layers": 1,
                },
            },
            "output_mode": "restoration",
        },
        "datasets": {
            "train": {
                "spike": {
                    "representation": "raw_window",
                    "raw_window_length": raw_window_length,
                    "reconstruction": {"type": "spikecv_tfp", "num_bins": 4},
                }
            }
        },
    }


def test_vrt_builds_with_dual_scale_temporal_mamba_raw_window_config():
    model = VRT(
        upscale=1,
        in_chans=24,
        out_chans=3,
        img_size=[6, 16, 16],
        window_size=[6, 8, 8],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=_dual_scale_raw_window_opt(),
    )

    assert model.fusion_operator is not None
    assert model.fusion_operator.spike_chans == 21
    assert model._fusion_spike_representation == "raw_window"
    assert model._fusion_raw_window_length == 21
    assert model.fusion_adapter.frame_contract == "collapsed"


def test_vrt_rejects_dual_scale_temporal_mamba_without_raw_window():
    bad_opt = _dual_scale_raw_window_opt()
    bad_opt["datasets"]["train"]["spike"]["representation"] = "tfp"

    with pytest.raises(ValueError, match="raw_window"):
        VRT(
            upscale=1,
            in_chans=7,
            out_chans=3,
            img_size=[6, 16, 16],
            window_size=[6, 8, 8],
            depths=[1] * 8,
            indep_reconsts=[],
            embed_dims=[16] * 8,
            num_heads=[1] * 8,
            pa_frames=2,
            use_flash_attn=False,
            optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
            opt=bad_opt,
        )


def test_vrt_dual_scale_temporal_mamba_records_representation_metadata(monkeypatch):
    model = VRT(
        upscale=1,
        in_chans=24,
        out_chans=3,
        img_size=[6, 8, 8],
        window_size=[6, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=_dual_scale_raw_window_opt(),
    )

    monkeypatch.setattr(model.fusion_adapter.operator, "forward", lambda rgb, spike: rgb)
    monkeypatch.setattr(
        model.fusion_adapter.operator,
        "diagnostics",
        lambda: {"local_norm": 1.0, "global_norm": 2.0, "summary_gate_mean": 0.5, "warmup_stage": "full"},
    )
    dummy_flows = [torch.zeros(1, 5, 2, 8, 8)] * 4
    monkeypatch.setattr(model, "get_flows", lambda _x, flow_spike=None: (dummy_flows, dummy_flows))
    monkeypatch.setattr(
        model,
        "get_aligned_image_2frames",
        lambda _x, _fb, _ff: [torch.zeros(1, 6, model.backbone_in_chans * 4, 8, 8)] * 2,
    )
    monkeypatch.setattr(model, "forward_features", lambda _x, *_args, **_kwargs: torch.zeros_like(_x))

    x = torch.randn(1, 6, 24, 8, 8)
    with torch.no_grad():
        _ = model(x)

    assert model._last_fusion_meta["spike_representation"] == "raw_window"
    assert model._last_fusion_meta["spike_window_length"] == 21
    assert model._last_fusion_meta["local_norm"] == 1.0
    assert model._last_fusion_meta["global_norm"] == 2.0
```

- [ ] **Step 2: Run the focused VRT tests and confirm they fail**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py -k "dual_scale_temporal_mamba" -v
```

Expected: FAIL because VRT does not yet allow raw-window input for the new operator and still constructs non-`pase_residual` Mamba-family operators with `spike_chans=1`.

- [ ] **Step 3: Update VRT legality and spike-channel resolution**

Apply these focused edits to `models/architectures/vrt/vrt.py`:

```python
            if normalized_operator_name in {'mamba', 'attention'}:
                if fusion_placement != 'early':
                    raise ValueError(f"fusion.operator='{normalized_operator_name}' requires fusion.placement='early'.")
                if early_out_chans != 3:
                    raise ValueError(f"fusion.operator='{normalized_operator_name}' requires fusion.out_chans=3 for early fusion.")
                if bool(early_cfg.get('expand_to_full_t', False)) and effective_frame_contract == 'collapsed':
                    raise ValueError(
                        f"fusion.operator='{normalized_operator_name}' does not support fusion.early.expand_to_full_t=true."
                    )
            if normalized_operator_name == 'dual_scale_temporal_mamba':
                if fusion_placement != 'early':
                    raise ValueError("fusion.operator='dual_scale_temporal_mamba' requires fusion.placement='early'.")
                if early_out_chans != 3:
                    raise ValueError("fusion.operator='dual_scale_temporal_mamba' requires fusion.out_chans=3 for early fusion.")
                if effective_frame_contract != 'collapsed':
                    raise ValueError("fusion.operator='dual_scale_temporal_mamba' requires fusion.early.frame_contract='collapsed'.")
                if spike_repr != 'raw_window':
                    raise ValueError(
                        "fusion.operator='dual_scale_temporal_mamba' requires spike.representation='raw_window'."
                    )
```

```python
            early_operator_spike_chans = (
                spike_input_chans
                if normalized_operator_name in {'pase_residual', 'dual_scale_temporal_mamba'}
                else 1
            )
```

```python
                if self._fusion_spike_representation is not None:
                    self._last_fusion_meta['spike_representation'] = self._fusion_spike_representation
                    if self._fusion_spike_representation == 'raw_window':
                        self._last_fusion_meta['spike_window_length'] = self._fusion_raw_window_length
```

Keep the existing metadata merge path intact so operator diagnostics continue to flow through `EarlyFusionAdapter._attach_operator_diagnostics()`.

- [ ] **Step 4: Re-run the focused VRT tests and confirm they pass**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py -k "dual_scale_temporal_mamba" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/architectures/vrt/vrt.py tests/models/test_vrt_fusion_integration.py
git commit -m "feat(vrt): wire dual-scale temporal mamba fusion"
```

## Task 4: Add a Runnable Budget-Controlled Config and Config-Level Coverage

**Files:**
- Create: `options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json`
- Modify: `tests/e2e/test_gopro_rgbspike_server_e2e.py`

- [ ] **Step 1: Write the failing config test**

Append this test to `tests/e2e/test_gopro_rgbspike_server_e2e.py`:

```python
from utils import utils_option as option


def test_dual_scale_temporal_mamba_raw_window_config_parses():
    opt = option.parse("options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json", is_train=True)

    assert opt["netG"]["fusion"]["operator"] == "dual_scale_temporal_mamba"
    assert opt["datasets"]["train"]["spike"]["representation"] == "raw_window"
    assert opt["datasets"]["test"]["spike"]["representation"] == "raw_window"
    assert opt["datasets"]["train"]["spike_channels"] == 21
    assert opt["netG"]["input"]["raw_ingress_chans"] == 24
```

- [ ] **Step 2: Run the config test and confirm it fails**

Run:

```bash
python -m pytest tests/e2e/test_gopro_rgbspike_server_e2e.py -k "dual_scale_temporal_mamba_raw_window_config_parses" -v
```

Expected: FAIL with `No such file or directory` for the new option file.

- [ ] **Step 3: Create the first runnable config**

Create `options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json` by copying `options/gopro_rgbspike_server_pase_residual_raw_window.json` and applying these targeted edits:

```json
"task": "gopro_raw21_scflow_dual_scale_temporal_mamba"
```

```json
"spike_channels": 21
```

```json
"raw_window_length": 21
```

```json
"raw_ingress_chans": 24
```

```json
"operator": "dual_scale_temporal_mamba"
```

```json
"operator_params": {
  "token_dim": 48,
  "patch_stride": 4,
  "d_state": 32,
  "d_conv": 4,
  "expand": 2,
  "local_layers": 1,
  "global_layers": 1,
  "alpha_init": 0.05,
  "gate_bias_init": -2.0,
  "enable_diagnostics": true
}
```

```json
"wandb_name": "gopro_raw21_scflow_dual_scale_temporal_mamba"
```

```json
"swanlab_name": "gopro_raw21_scflow_dual_scale_temporal_mamba"
```

```json
"swanlab_description": "early fusion dual-scale temporal mamba + raw_window + scflow"
```

- [ ] **Step 4: Re-run the config test and confirm it passes**

Run:

```bash
python -m pytest tests/e2e/test_gopro_rgbspike_server_e2e.py -k "dual_scale_temporal_mamba_raw_window_config_parses" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json tests/e2e/test_gopro_rgbspike_server_e2e.py
git commit -m "chore(options): add dual-scale temporal mamba raw-window config"
```

## Task 5: Run Focused Regressions and Record the Hand-off Surface

**Files:**
- Modify: `docs/superpowers/plans/2026-04-30-dual-scale-temporal-mamba-implementation.md`

- [ ] **Step 1: Run the focused regression suite**

Run:

```bash
python -m pytest tests/models/test_dual_scale_temporal_mamba.py tests/models/test_vrt_fusion_integration.py -k "dual_scale_temporal_mamba" -v
```

Expected: PASS

Run:

```bash
python -m pytest tests/data/test_spike_raw_window.py tests/data/test_dataset_rgbspike_raw_window.py tests/e2e/test_gopro_rgbspike_server_e2e.py -k "raw_window or dual_scale_temporal_mamba_raw_window_config_parses" -v
```

Expected: PASS

- [ ] **Step 2: Smoke-check the parsed option file**

Run:

```bash
python -c "from utils import utils_option as option; opt = option.parse('options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json', is_train=True); print(opt['task']); print(opt['netG']['fusion']['operator']); print(opt['datasets']['train']['spike']['raw_window_length']); print(opt['netG']['input']['raw_ingress_chans'])"
```

Expected output:

```text
gopro_raw21_scflow_dual_scale_temporal_mamba
dual_scale_temporal_mamba
21
24
```

- [ ] **Step 3: Commit the verified implementation batch**

```bash
git add models/fusion/operators/dual_scale_temporal_mamba.py models/fusion/operators/__init__.py models/architectures/vrt/vrt.py tests/models/test_dual_scale_temporal_mamba.py tests/models/test_vrt_fusion_integration.py options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json tests/e2e/test_gopro_rgbspike_server_e2e.py
git commit -m "test(fusion): verify dual-scale temporal mamba raw-window wiring"
```

## Self-Review

- Spec coverage: this plan covers the new operator, VRT legality checks, raw-window reuse, metadata surfacing, a runnable config, and the baseline regression surface described in the spec.
- Placeholder scan: no `TODO`, `TBD`, or deferred code placeholders remain in the tasks.
- Type consistency: the plan uses `dual_scale_temporal_mamba`, `raw_window`, `raw_window_length`, `local_layers`, `global_layers`, `local_norm`, and `global_norm` consistently across operator, VRT, tests, and config tasks.
