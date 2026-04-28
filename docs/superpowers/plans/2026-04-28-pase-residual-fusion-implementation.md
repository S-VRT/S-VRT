# PASE-Residual Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PASE-Residual Fusion` as a collapsed-only strong non-sequential early-fusion baseline that lets PASE consume the full per-frame spike clip and uses the stabilized residual write-back head.

**Architecture:** Introduce a new structured early operator `pase_residual` that flattens `[B, T, S, H, W]` to `[B*T, S, H, W]`, runs `PixelAdaptiveSpikeEncoder` over the full `S` bins, fuses it with an RGB context encoder, and writes back a conservative RGB residual with `delta`, `gate`, and `alpha`. Wire VRT so this operator receives the real spike-bin width instead of the old early-operator default `1`, reject expanded contract usage, and extend training logs to record the new diagnostics.

**Tech Stack:** Python, PyTorch, pytest, existing VRT/fusion adapter stack

---

## File Map

- Create: `models/fusion/operators/pase_residual.py`
  - New collapsed structured early-fusion operator.
  - Named submodules for warmup staging: `rgb_context_encoder`, `pase`, `fusion_body`, `fusion_writeback_head`, `alpha`.
  - `set_warmup_stage()` and `diagnostics()` helpers.
- Modify: `models/fusion/operators/__init__.py`
  - Register `pase_residual` in `build_operator()` and `__all__`.
- Modify: `models/architectures/vrt/vrt.py`
  - Pass full spike-bin width into `pase_residual` instead of hard-coded early `spike_chans=1`.
  - Reject `pase_residual` under `expanded` frame contract.
  - Keep `pase_residual` constrained to early placement with RGB residual output width `3`.
- Modify: `models/model_plain.py`
  - Extend `_record_fusion_diagnostics_to_log()` to export `pase_norm` and `fusion_body_norm`.
- Modify: `tests/models/test_fusion_early_adapter.py`
  - Lock operator semantics, startup behavior, and warmup freezing.
- Modify: `tests/models/test_vrt_fusion_integration.py`
  - Lock VRT build semantics, constructor spike width, and expanded-contract rejection.
- Modify: `tests/models/test_two_phase_training.py`
  - Lock training-log propagation of `pase_residual` diagnostics.
- Create: `options/gopro_rgbspike_server_pase_residual.json`
  - Runnable ablation config that swaps only the fusion operator family and keeps collapsed semantics explicit.

### Task 1: Lock `PASE-Residual Fusion` Operator Semantics in Unit Tests

**Files:**
- Modify: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: Write the failing operator tests**

Update the frame-contract metadata parameterization and add these tests to `tests/models/test_fusion_early_adapter.py`:

```python
@pytest.mark.parametrize(
    ("operator_name", "rgb_chans", "spike_chans", "expected_frame_contract"),
    [
        ("gated", 3, 1, "expanded"),
        ("concat", 3, 1, "expanded"),
        ("pase", 3, 1, "expanded"),
        ("mamba", 3, 1, "collapsed"),
        ("pase_residual", 3, 4, "collapsed"),
    ],
)
def test_fusion_operator_frame_contract_metadata(operator_name, rgb_chans, spike_chans, expected_frame_contract):
    op = create_fusion_operator(operator_name, rgb_chans, spike_chans, 3, {})
    assert getattr(op, "frame_contract", None) == expected_frame_contract


def test_pase_residual_operator_shape():
    op = create_fusion_operator(
        "pase_residual",
        3,
        4,
        3,
        {"feature_chans": 16, "hidden_chans": 8},
    )
    rgb_feat = torch.randn(2, 5, 3, 12, 12)
    spike_feat = torch.randn(2, 5, 4, 12, 12)
    out = op(rgb_feat, spike_feat)
    assert out.shape == (2, 5, 3, 12, 12)


def test_pase_residual_operator_small_output_non_degenerate_init():
    op = create_fusion_operator(
        "pase_residual",
        3,
        4,
        3,
        {
            "feature_chans": 12,
            "hidden_chans": 8,
            "alpha_init": 0.05,
            "gate_bias_init": -2.0,
            "enable_diagnostics": True,
        },
    )
    rgb = torch.ones(1, 2, 3, 8, 8) * 0.5
    spike = torch.zeros(1, 2, 4, 8, 8)

    out = op(rgb, spike)

    max_diff = (out - rgb).abs().max().item()
    diagnostics = op.diagnostics()
    assert 1e-7 < max_diff < 1e-2
    assert 1e-7 < diagnostics["effective_update_norm"] < 1e-2
    assert diagnostics["warmup_stage"] == "full"
    assert set(diagnostics) >= {
        "pase_norm",
        "fusion_body_norm",
        "delta_norm",
        "gate_mean",
        "effective_update_norm",
        "warmup_stage",
    }

    loss = (out - rgb).abs().mean()
    loss.backward()

    delta_grad = op.fusion_writeback_head["delta"].weight.grad
    gate_grad = op.fusion_writeback_head["gate"].weight.grad
    assert delta_grad is not None and delta_grad.abs().sum().item() > 0.0
    assert gate_grad is not None and gate_grad.abs().sum().item() > 0.0


def test_pase_residual_operator_writeback_only_stage_freezes_feature_paths():
    op = create_fusion_operator(
        "pase_residual",
        3,
        4,
        3,
        {"feature_chans": 16, "hidden_chans": 8},
    )

    op.set_warmup_stage("writeback_only")

    assert all(not p.requires_grad for p in op.rgb_context_encoder.parameters())
    assert all(not p.requires_grad for p in op.pase.parameters())
    assert all(not p.requires_grad for p in op.fusion_body.parameters())
    assert all(p.requires_grad for p in op.fusion_writeback_head.parameters())
    assert op.alpha.requires_grad is True


def test_pase_residual_operator_token_mixer_stage_unfreezes_all_feature_paths():
    op = create_fusion_operator(
        "pase_residual",
        3,
        4,
        3,
        {"feature_chans": 16, "hidden_chans": 8},
    )

    op.set_warmup_stage("token_mixer")

    assert all(p.requires_grad for p in op.rgb_context_encoder.parameters())
    assert all(p.requires_grad for p in op.pase.parameters())
    assert all(p.requires_grad for p in op.fusion_body.parameters())
    assert all(p.requires_grad for p in op.fusion_writeback_head.parameters())
    assert op.alpha.requires_grad is True
```

- [ ] **Step 2: Run the focused operator tests to verify they fail**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py -k "pase_residual" -v
```

Expected: FAIL with `Unknown fusion operator: pase_residual`.

- [ ] **Step 3: Implement the new operator and registry plumbing**

Create `models/fusion/operators/pase_residual.py` with this implementation:

```python
from typing import Dict

import torch
from torch import nn

from models.spk_encoder import PixelAdaptiveSpikeEncoder


class PaseResidualFusionOperator(nn.Module):
    expects_structured_early = True
    frame_contract = "collapsed"

    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int, operator_params: Dict):
        super().__init__()
        if rgb_chans != 3:
            raise ValueError("PaseResidualFusionOperator requires rgb_chans=3.")
        if out_chans != 3:
            raise ValueError("PaseResidualFusionOperator requires out_chans=3.")
        if spike_chans <= 0:
            raise ValueError("PaseResidualFusionOperator requires spike_chans > 0.")

        feature_chans = int(operator_params.get("feature_chans", operator_params.get("hidden_chans", 48)))
        pase_hidden_chans = int(operator_params.get("pase_hidden_chans", operator_params.get("hidden_chans", 32)))
        kernel_size = int(operator_params.get("kernel_size", 3))
        normalize_kernel = bool(operator_params.get("normalize_kernel", True))
        alpha_init = float(operator_params.get("alpha_init", 0.05))
        gate_bias_init = float(operator_params.get("gate_bias_init", -2.0))
        enable_diagnostics = bool(operator_params.get("enable_diagnostics", False))

        self.rgb_chans = rgb_chans
        self.spike_chans = spike_chans
        self.out_chans = out_chans
        self.enable_diagnostics = enable_diagnostics

        self.rgb_context_encoder = nn.Sequential(
            nn.Conv2d(3, feature_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_chans, feature_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pase = PixelAdaptiveSpikeEncoder(
            in_chans=spike_chans,
            out_chans=feature_chans,
            kernel_size=kernel_size,
            hidden_chans=pase_hidden_chans,
            normalize_kernel=normalize_kernel,
        )
        self.fusion_body = nn.Sequential(
            nn.Conv2d(feature_chans * 2, feature_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_chans, feature_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fusion_writeback_head = nn.ModuleDict(
            {
                "delta": nn.Conv2d(feature_chans, 3, kernel_size=1),
                "gate": nn.Conv2d(feature_chans, 3, kernel_size=1),
            }
        )
        self.alpha = nn.Parameter(torch.full((1, 3, 1, 1), alpha_init))
        self._warmup_stage = "full"
        self._last_diagnostics = {"warmup_stage": "full"}

        nn.init.normal_(self.fusion_writeback_head["delta"].weight, std=1e-3)
        nn.init.zeros_(self.fusion_writeback_head["delta"].bias)
        nn.init.normal_(self.fusion_writeback_head["gate"].weight, std=1e-3)
        nn.init.constant_(self.fusion_writeback_head["gate"].bias, gate_bias_init)

    def set_warmup_stage(self, stage) -> None:
        normalized = "full" if stage in {None, "", "full"} else str(stage).strip().lower()
        if normalized not in {"full", "writeback_only", "token_mixer"}:
            raise ValueError(f"Unsupported PASE warmup stage: {stage!r}")
        self._warmup_stage = normalized
        feature_trainable = normalized != "writeback_only"
        for module in (self.rgb_context_encoder, self.pase, self.fusion_body):
            for param in module.parameters():
                param.requires_grad_(feature_trainable)
        for param in self.fusion_writeback_head.parameters():
            param.requires_grad_(True)
        self.alpha.requires_grad_(True)

    def diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() != 5:
            raise ValueError("pase_residual early fusion expects rgb with shape [B, T, 3, H, W].")
        if spike_feat.dim() != 5:
            raise ValueError("pase_residual early fusion expects spike with shape [B, T, S, H, W].")

        bsz, steps, rgb_chans, height, width = rgb_feat.shape
        spike_bsz, spike_steps, spike_chans, spike_height, spike_width = spike_feat.shape
        if (bsz, steps, height, width) != (spike_bsz, spike_steps, spike_height, spike_width):
            raise ValueError("rgb and spike must share batch, time, height, and width dimensions")
        if rgb_chans != self.rgb_chans:
            raise ValueError(f"Expected rgb channels={self.rgb_chans}, got {rgb_chans}")
        if spike_chans != self.spike_chans:
            raise ValueError(f"Expected spike channels={self.spike_chans}, got {spike_chans}")

        rgb_flat = rgb_feat.reshape(bsz * steps, rgb_chans, height, width)
        spike_flat = spike_feat.reshape(bsz * steps, spike_chans, height, width)

        rgb_ctx = self.rgb_context_encoder(rgb_flat)
        pase_feat = self.pase(spike_flat)
        fused_feat = self.fusion_body(torch.cat([rgb_ctx, pase_feat], dim=1))
        delta = self.fusion_writeback_head["delta"](fused_feat)
        gate = torch.sigmoid(self.fusion_writeback_head["gate"](fused_feat))
        effective_update = self.alpha.view(1, 3, 1, 1) * gate * delta
        out = rgb_flat + effective_update

        if self.enable_diagnostics:
            self._last_diagnostics = {
                "pase_norm": float(pase_feat.detach().float().norm(dim=1).mean().item()),
                "fusion_body_norm": float(fused_feat.detach().float().norm(dim=1).mean().item()),
                "delta_norm": float(delta.detach().float().abs().mean().item()),
                "gate_mean": float(gate.detach().float().mean().item()),
                "effective_update_norm": float(effective_update.detach().float().abs().mean().item()),
                "warmup_stage": self._warmup_stage,
            }
        else:
            self._last_diagnostics = {"warmup_stage": self._warmup_stage}

        return out.reshape(bsz, steps, 3, height, width)


__all__ = ["PaseResidualFusionOperator"]
```

Update `models/fusion/operators/__init__.py`:

```python
from .pase_residual import PaseResidualFusionOperator
```

```python
    if normalized_name == 'pase_residual':
        return PaseResidualFusionOperator(
            rgb_chans=rgb_chans,
            spike_chans=spike_chans,
            out_chans=out_chans,
            operator_params=operator_params,
        )
```

```python
    'PaseResidualFusionOperator',
```

- [ ] **Step 4: Run the focused operator tests to verify they pass**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py -k "pase_residual" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion/operators/pase_residual.py models/fusion/operators/__init__.py tests/models/test_fusion_early_adapter.py
git commit -m "feat(fusion): add PASE residual baseline operator"
```

### Task 2: Wire `pase_residual` Through VRT as a Collapsed-Only Early Operator

**Files:**
- Modify: `models/architectures/vrt/vrt.py`
- Modify: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Write the failing VRT integration tests**

Add these tests to `tests/models/test_vrt_fusion_integration.py`:

```python
def test_vrt_builds_with_pase_residual_collapsed_fusion_config():
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
                "operator": "pase_residual",
                "out_chans": 3,
                "early": {
                    "frame_contract": "collapsed",
                },
                "operator_params": {
                    "feature_chans": 16,
                    "hidden_chans": 8,
                },
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

    assert model.fusion_adapter.frame_contract == "collapsed"
    assert model.fusion_operator.spike_chans == 4


def test_vrt_rejects_pase_residual_expanded_frame_contract():
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
                "operator": "pase_residual",
                "out_chans": 3,
                "early": {
                    "frame_contract": "expanded",
                    "expand_to_full_t": True,
                },
                "operator_params": {},
            },
        }
    }

    with pytest.raises(ValueError, match="pase_residual"):
        VRT(
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
```

- [ ] **Step 2: Run the focused VRT tests to verify they fail**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py -k "pase_residual" -v
```

Expected: FAIL because VRT still passes `spike_chans=1` to all early operators and does not reject expanded `pase_residual`.

- [ ] **Step 3: Implement VRT validation and constructor spike-width routing**

Update `models/architectures/vrt/vrt.py` near the early-fusion operator creation logic:

```python
            early_operator_spike_chans = 1
            if normalized_operator_name == "pase_residual":
                if fusion_placement != "early":
                    raise ValueError("fusion.operator='pase_residual' requires fusion.placement='early'.")
                if early_out_chans != 3:
                    raise ValueError("fusion.operator='pase_residual' requires fusion.out_chans=3.")
                if effective_frame_contract != "collapsed":
                    raise ValueError("fusion.operator='pase_residual' requires fusion.early.frame_contract='collapsed'.")
                early_operator_spike_chans = spike_input_chans

            if fusion_placement == 'early':
                self.fusion_operator = create_fusion_operator(
                    operator_name=operator_name,
                    rgb_chans=3,
                    spike_chans=early_operator_spike_chans,
                    out_chans=early_out_chans,
                    operator_params=operator_params,
                )
```

Do not change the adapter-side `spike_chans=spike_input_chans` argument. That existing path is still needed for structured spike upsampling and metadata.

- [ ] **Step 4: Run the focused VRT tests to verify they pass**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py -k "pase_residual" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/architectures/vrt/vrt.py tests/models/test_vrt_fusion_integration.py
git commit -m "feat(vrt): wire PASE residual fusion as collapsed early operator"
```

### Task 3: Export `PASE-Residual Fusion` Diagnostics to Training Logs

**Files:**
- Modify: `models/model_plain.py`
- Modify: `tests/models/test_two_phase_training.py`

- [ ] **Step 1: Write the failing diagnostics-log test**

Add this test to `tests/models/test_two_phase_training.py` after the existing fusion diagnostics log test:

```python
def test_model_plain_current_log_includes_selected_pase_residual_diagnostics():
    from collections import OrderedDict
    from models.model_plain import ModelPlain

    model = ModelPlain.__new__(ModelPlain)
    model.log_dict = OrderedDict([("G_loss", 1.0)])
    model.timer = type("TimerStub", (), {"get_current_timings": staticmethod(lambda: {})})()

    model._record_fusion_diagnostics_to_log(
        {
            "pase_norm": 1.25,
            "fusion_body_norm": 0.8,
            "delta_norm": 0.02,
            "gate_mean": 0.3,
            "effective_update_norm": 0.01,
            "warmup_stage": "token_mixer",
        }
    )

    log = model.current_log()
    assert log["fusion_pase_norm"] == 1.25
    assert log["fusion_fusion_body_norm"] == 0.8
    assert log["fusion_gate_mean"] == 0.3
    assert log["fusion_warmup_stage"] == "token_mixer"
```

- [ ] **Step 2: Run the focused diagnostics-log test to verify it fails**

Run:

```bash
python -m pytest tests/models/test_two_phase_training.py::test_model_plain_current_log_includes_selected_pase_residual_diagnostics -v
```

Expected: FAIL with missing `fusion_pase_norm` and `fusion_fusion_body_norm`.

- [ ] **Step 3: Extend `_record_fusion_diagnostics_to_log()` for the new diagnostics**

Update `models/model_plain.py`:

```python
    def _record_fusion_diagnostics_to_log(self, fusion_meta) -> None:
        if not isinstance(fusion_meta, dict):
            return
        if "warmup_stage" in fusion_meta:
            self.log_dict["fusion_warmup_stage"] = fusion_meta["warmup_stage"]
        for key in (
            "token_norm",
            "mamba_norm",
            "pase_norm",
            "fusion_body_norm",
            "delta_norm",
            "gate_mean",
            "effective_update_norm",
        ):
            if key in fusion_meta:
                self.log_dict[f"fusion_{key}"] = float(fusion_meta[key])
```

Keep the existing behavior for `mamba` diagnostics intact; just extend the stable key set.

- [ ] **Step 4: Run the focused diagnostics-log test to verify it passes**

Run:

```bash
python -m pytest tests/models/test_two_phase_training.py::test_model_plain_current_log_includes_selected_pase_residual_diagnostics -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/model_plain.py tests/models/test_two_phase_training.py
git commit -m "feat(train): log PASE residual fusion diagnostics"
```

### Task 4: Add a Runnable Ablation Config for `PASE-Residual Fusion`

**Files:**
- Create: `options/gopro_rgbspike_server_pase_residual.json`

- [ ] **Step 1: Create the dedicated ablation config**

Create `options/gopro_rgbspike_server_pase_residual.json` by copying the current server config and changing the task name plus fusion block to:

```json
"task": "gopro_tfp4_scflow_pase_residual"
```

```json
"fusion": {
  "enable": true,
  "placement": "early",
  "operator": "pase_residual",
  "mode": "replace",
  "out_chans": 3,
  "operator_params": {
    "feature_chans": 48,
    "hidden_chans": 32,
    "pase_hidden_chans": 32,
    "kernel_size": 3,
    "normalize_kernel": true,
    "alpha_init": 0.05,
    "gate_bias_init": -2.0,
    "enable_diagnostics": true
  },
  "early": {
    "frame_contract": "collapsed"
  }
}
```

Keep the rest of the training setup unchanged so the ablation isolates the fusion operator family.

- [ ] **Step 2: Validate the config parses**

Run:

```bash
python -c "from utils import utils_option as option; opt = option.parse('options/gopro_rgbspike_server_pase_residual.json', is_train=True); print(opt['task']); print(opt['netG']['fusion']['operator']); print(opt['netG']['fusion']['early']['frame_contract'])"
```

Expected output:

```text
gopro_tfp4_scflow_pase_residual
pase_residual
collapsed
```

- [ ] **Step 3: Commit**

```bash
git add options/gopro_rgbspike_server_pase_residual.json
git commit -m "chore(options): add PASE residual fusion ablation config"
```

### Task 5: Run the Focused Regression Slice for the New Baseline

**Files:**
- Modify: none
- Test: `tests/models/test_fusion_early_adapter.py`
- Test: `tests/models/test_vrt_fusion_integration.py`
- Test: `tests/models/test_two_phase_training.py`

- [ ] **Step 1: Run the combined regression slice**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py tests/models/test_vrt_fusion_integration.py tests/models/test_two_phase_training.py -k "pase_residual" -v
```

Expected: PASS

- [ ] **Step 2: Run the server config parser check once more**

Run:

```bash
python -c "from utils import utils_option as option; opt = option.parse('options/gopro_rgbspike_server_pase_residual.json', is_train=True); print(opt['netG']['fusion']['operator'])"
```

Expected output:

```text
pase_residual
```

- [ ] **Step 3: Commit the verification checkpoint**

```bash
git commit --allow-empty -m "test(fusion): verify PASE residual baseline wiring"
```
