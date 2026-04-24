# Fusion Mamba Tokenized Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current early-fusion `mamba` path with a tokenized, diagnostically visible operator plus staged phase-1 warmup so fusion starts learning early instead of stalling in a near-zero residual regime.

**Architecture:** Keep `mamba` as a collapsed early operator, but move spike-bin sequence modeling from full-resolution per-pixel paths to low-resolution token grids. Add a write-back head with explicit residual scale `alpha`, teach `ModelPlain` to stage phase-1 trainability and loss weights, and surface operator diagnostics through the existing fusion metadata path.

**Tech Stack:** Python, PyTorch, pytest, existing VRT/fusion adapter stack

---

## File Map

- Modify: `models/fusion/operators/mamba.py`
  - Replace per-pixel sequence mixing with tokenized low-resolution mixing.
  - Add named submodules for warmup staging: `rgb_context_encoder`, `spike_token_encoder`, `mamba_token_mixer`, `fusion_writeback_head`, `alpha`.
  - Add `set_warmup_stage()` and `diagnostics()` helpers.
  - Preserve backward compatibility for legacy config keys `model_dim` and `init_gate_bias`.
- Modify: `models/model_plain.py`
  - Keep frozen parameters inside the optimizer so later warmup unfreezing actually updates them.
  - Add phase-1 warmup stage resolution and handoff into the fusion operator.
  - Add scheduled passthrough weight and update-penalty computation.
  - Copy a stable subset of fusion diagnostics into `log_dict` so early-phase behavior is visible in normal training dashboards.
- Modify: `models/fusion/adapters/early.py`
  - Merge operator diagnostics into `result["meta"]` for both phase-1 fast-path and full VRT forward.
- Modify: `tests/models/test_fusion_early_adapter.py`
  - Lock tokenized `mamba` diagnostics and warmup-stage behavior.
- Modify: `tests/models/test_two_phase_training.py`
  - Lock `ModelPlain` warmup-stage handoff and optimizer coverage for later unfreeze.
- Modify: `tests/models/test_model_plain_fusion_aux_loss.py`
  - Lock phase-1 passthrough decay and update-penalty behavior.
- Modify: `tests/models/test_vrt_fusion_integration.py`
  - Verify diagnostics survive the adapter/VRT cache path for collapsed early `mamba`.
- Modify: `tests/models/test_fusion_debug_dumper.py`
  - Lock the chosen phase-1 debug path for collapsed early `mamba` so runnable configs do not regress back to CPU full-frame replay.
- Modify: `options/gopro_rgbspike_server.json`
  - Add explicit tokenized `mamba` params, `train.fusion_warmup`, and switch tokenized `mamba` debug dumping to the train-batch path.
- Modify: `options/gopro_rgbspike_server_debug.json`
  - Keep the debug run aligned with the main server config.

### Task 1: Rebuild `MambaFusionOperator` around tokenized low-resolution mixing

**Files:**
- Modify: `models/fusion/operators/mamba.py`
- Test: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: Write the failing operator tests**

Add these tests to `tests/models/test_fusion_early_adapter.py`:

```python
def test_mamba_operator_exposes_scalar_diagnostics_or_missing_dep():
    op = create_fusion_operator(
        "mamba",
        3,
        1,
        3,
        {
            "token_dim": 24,
            "token_stride": 2,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "num_layers": 1,
            "alpha_init": 0.05,
            "gate_bias_init": -2.0,
            "enable_diagnostics": True,
        },
    )
    rgb = torch.randn(1, 2, 3, 12, 12)
    spike = torch.randn(1, 2, 6, 12, 12)

    try:
        _ = op(rgb, spike)
    except RuntimeError as exc:
        assert "mamba_ssm is required" in str(exc)
        return

    diagnostics = op.diagnostics()
    assert diagnostics["warmup_stage"] == "full"
    assert set(diagnostics) >= {
        "token_norm",
        "mamba_norm",
        "delta_norm",
        "gate_mean",
        "effective_update_norm",
        "warmup_stage",
    }
    assert all(isinstance(diagnostics[key], float) for key in diagnostics if key != "warmup_stage")


def test_mamba_operator_small_output_non_degenerate_init_or_missing_dep():
    op = create_fusion_operator(
        "mamba",
        3,
        1,
        3,
        {
            "token_dim": 24,
            "token_stride": 2,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "num_layers": 1,
            "alpha_init": 0.05,
            "gate_bias_init": -2.0,
            "enable_diagnostics": True,
        },
    )
    rgb = torch.ones(1, 2, 3, 8, 8) * 0.5
    spike = torch.zeros(1, 2, 6, 8, 8)

    try:
        out = op(rgb, spike)
    except RuntimeError as exc:
        assert "mamba_ssm is required" in str(exc)
        return

    max_diff = (out - rgb).abs().max().item()
    diagnostics = op.diagnostics()
    assert 1e-7 < max_diff < 1e-2
    assert 1e-7 < diagnostics["effective_update_norm"] < 1e-2

    loss = (out - rgb).abs().mean()
    loss.backward()

    delta_grad = op.fusion_writeback_head["delta"].weight.grad
    gate_grad = op.fusion_writeback_head["gate"].weight.grad
    assert delta_grad is not None and delta_grad.abs().sum().item() > 0.0
    assert gate_grad is not None and gate_grad.abs().sum().item() > 0.0


def test_mamba_operator_writeback_only_stage_freezes_token_mixers():
    op = create_fusion_operator(
        "mamba",
        3,
        1,
        3,
        {"token_dim": 16, "token_stride": 2, "num_layers": 1},
    )

    op.set_warmup_stage("writeback_only")

    assert all(not p.requires_grad for p in op.rgb_context_encoder.parameters())
    assert all(not p.requires_grad for p in op.spike_token_encoder.parameters())
    assert all(not p.requires_grad for p in op.mamba_token_mixer.parameters())
    assert all(p.requires_grad for p in op.fusion_writeback_head.parameters())
    assert op.alpha.requires_grad is True


def test_mamba_operator_token_mixer_stage_unfreezes_temporal_stack():
    op = create_fusion_operator(
        "mamba",
        3,
        1,
        3,
        {"token_dim": 16, "token_stride": 2, "num_layers": 1},
    )

    op.set_warmup_stage("token_mixer")

    assert all(p.requires_grad for p in op.rgb_context_encoder.parameters())
    assert all(p.requires_grad for p in op.spike_token_encoder.parameters())
    assert all(p.requires_grad for p in op.mamba_token_mixer.parameters())
    assert all(p.requires_grad for p in op.fusion_writeback_head.parameters())
    assert op.alpha.requires_grad is True
```

Replace the current strict passthrough test `test_mamba_operator_passthrough_at_init_or_missing_dep()` with `test_mamba_operator_small_output_non_degenerate_init_or_missing_dep()`. The redesigned operator no longer promises exact identity at `atol=1e-5`; it promises small but non-zero output magnitude plus a non-degenerate learnable path with live gradients through the write-back head.

- [ ] **Step 2: Run the operator tests to verify they fail**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py -k "scalar_diagnostics or small_output_non_degenerate_init or writeback_only_stage or token_mixer_stage" -v
```

Expected: FAIL because `MambaFusionOperator` does not yet expose `diagnostics()`, `set_warmup_stage()`, or the named warmup submodules.

- [ ] **Step 3: Implement the tokenized operator**

Update `models/fusion/operators/mamba.py` so the operator has explicit token encoders, explicit write-back state, and warmup-aware parameter grouping:

```python
from typing import Dict

import torch
import torch.nn.functional as F
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
        if not tokens.is_cuda:
            raise RuntimeError("mamba_ssm is required for mamba fusion operator with CUDA tensors.")
        return tokens + self.mamba(self.norm(tokens))


class MambaFusionOperator(nn.Module):
    expects_structured_early = True
    frame_contract = "collapsed"

    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int, operator_params: Dict):
        super().__init__()
        if rgb_chans != 3:
            raise ValueError("MambaFusionOperator requires rgb_chans=3.")
        if spike_chans != 1:
            raise ValueError("MambaFusionOperator requires spike_chans=1 at construction time.")
        if out_chans != 3:
            raise ValueError("MambaFusionOperator requires out_chans=3.")

        token_dim = int(operator_params.get("token_dim", operator_params.get("model_dim", 48)))
        token_stride = int(operator_params.get("token_stride", 4))
        d_state = int(operator_params.get("d_state", 32))
        d_conv = int(operator_params.get("d_conv", 4))
        expand = int(operator_params.get("expand", 2))
        num_layers = int(operator_params.get("num_layers", 3))
        alpha_init = float(operator_params.get("alpha_init", 0.05))
        gate_bias_init = float(operator_params.get("gate_bias_init", operator_params.get("init_gate_bias", -2.0)))
        enable_diagnostics = bool(operator_params.get("enable_diagnostics", True))

        self.enable_diagnostics = enable_diagnostics
        self.rgb_context_encoder = nn.Sequential(
            nn.Conv2d(3, token_dim, kernel_size=3, stride=token_stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.spike_token_encoder = nn.Sequential(
            nn.Conv2d(1, token_dim, kernel_size=3, stride=token_stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
        )
        self.mamba_token_mixer = nn.ModuleList(
            [_MambaBlock(model_dim=token_dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)]
        )
        self.fusion_writeback_head = nn.ModuleDict(
            {
                "body": nn.Sequential(
                    nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ),
                "delta": nn.Conv2d(token_dim, 3, kernel_size=1),
                "gate": nn.Conv2d(token_dim, 3, kernel_size=1),
            }
        )
        self.alpha = nn.Parameter(torch.full((1, 3, 1, 1), alpha_init))
        self._warmup_stage = "full"
        self._last_diagnostics = {"warmup_stage": "full"}

        nn.init.normal_(self.fusion_writeback_head["delta"].weight, std=1e-3)
        nn.init.zeros_(self.fusion_writeback_head["delta"].bias)
        nn.init.normal_(self.fusion_writeback_head["gate"].weight, std=1e-3)
        nn.init.constant_(self.fusion_writeback_head["gate"].bias, gate_bias_init)

    def set_warmup_stage(self, stage: str | None) -> None:
        normalized = "full" if stage in {None, "", "full"} else str(stage).strip().lower()
        if normalized not in {"full", "writeback_only", "token_mixer"}:
            raise ValueError(f"Unsupported Mamba warmup stage: {stage!r}")
        self._warmup_stage = normalized
        token_trainable = normalized != "writeback_only"
        for module in (self.rgb_context_encoder, self.spike_token_encoder, self.mamba_token_mixer):
            for param in module.parameters():
                param.requires_grad_(token_trainable)
        for param in self.fusion_writeback_head.parameters():
            param.requires_grad_(True)
        self.alpha.requires_grad_(True)

    def diagnostics(self) -> dict[str, float | str]:
        return dict(self._last_diagnostics)

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
        rgb_low = self.rgb_context_encoder(rgb_flat)
        _, token_dim, token_h, token_w = rgb_low.shape
        rgb_low = rgb_low.reshape(bsz, steps, token_dim, token_h, token_w)

        spike_flat = spike_feat.reshape(bsz * steps * spike_bins, 1, height, width)
        spike_low = self.spike_token_encoder(spike_flat).reshape(bsz, steps, spike_bins, token_dim, token_h, token_w)

        spike_tokens = spike_low.permute(0, 1, 4, 5, 2, 3).reshape(bsz * steps * token_h * token_w, spike_bins, token_dim)
        rgb_tokens = rgb_low.permute(0, 1, 3, 4, 2).reshape(bsz * steps * token_h * token_w, 1, token_dim)
        seq = spike_tokens + rgb_tokens
        for block in self.mamba_token_mixer:
            seq = block(seq)

        pooled = seq.mean(dim=1).reshape(bsz, steps, token_h, token_w, token_dim).permute(0, 1, 4, 2, 3)
        fused_low = pooled + rgb_low

        writeback = self.fusion_writeback_head["body"](fused_low.reshape(bsz * steps, token_dim, token_h, token_w))
        delta_low = self.fusion_writeback_head["delta"](writeback)
        gate_logits_low = self.fusion_writeback_head["gate"](writeback)
        delta = F.interpolate(delta_low, size=(height, width), mode="bilinear", align_corners=False).reshape(bsz, steps, 3, height, width)
        gate_logits = F.interpolate(gate_logits_low, size=(height, width), mode="bilinear", align_corners=False).reshape(bsz, steps, 3, height, width)
        gate = torch.sigmoid(gate_logits)
        effective_update = self.alpha.view(1, 1, 3, 1, 1) * gate * delta
        out = rgb_feat + effective_update

        if self.enable_diagnostics:
            self._last_diagnostics = {
                "token_norm": float(spike_tokens.detach().float().norm(dim=-1).mean().item()),
                "mamba_norm": float(seq.detach().float().norm(dim=-1).mean().item()),
                "delta_norm": float(delta.detach().float().abs().mean().item()),
                "gate_mean": float(gate.detach().float().mean().item()),
                "effective_update_norm": float(effective_update.detach().float().abs().mean().item()),
                "warmup_stage": self._warmup_stage,
            }
        else:
            self._last_diagnostics = {"warmup_stage": self._warmup_stage}
        return out
```

- [ ] **Step 4: Run the operator tests to verify they pass**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py -k "scalar_diagnostics or small_output_non_degenerate_init or writeback_only_stage or token_mixer_stage" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion/operators/mamba.py tests/models/test_fusion_early_adapter.py
git commit -m "feat(fusion): redesign mamba as tokenized early operator"
```

### Task 2: Teach `ModelPlain` to hand off phase-1 warmup stages and keep frozen params in the optimizer

**Files:**
- Modify: `models/model_plain.py`
- Test: `tests/models/test_two_phase_training.py`

- [ ] **Step 1: Write the failing warmup/optimizer tests**

Add these tests to `tests/models/test_two_phase_training.py`:

```python
from types import SimpleNamespace


class _WarmupAwareOperator:
    def __init__(self):
        self.last_stage = None

    def set_warmup_stage(self, stage):
        self.last_stage = stage


class _WarmupAwareNet:
    def __init__(self):
        self.fusion_enabled = True
        self.fusion_cfg = {"placement": "early"}
        self.fusion_adapter = SimpleNamespace(operator=_WarmupAwareOperator())


def test_model_plain_configures_writeback_only_stage_during_head_only_iters():
    from models.model_plain import ModelPlain

    model = ModelPlain.__new__(ModelPlain)
    model.opt_train = {"fusion_warmup": {"head_only_iters": 2}}
    model.fix_iter = 10
    model.netG = _WarmupAwareNet()

    stage = model._configure_fusion_warmup_trainability(current_step=0)

    assert stage == "writeback_only"
    assert model.netG.fusion_adapter.operator.last_stage == "writeback_only"


def test_model_plain_switches_to_token_mixer_stage_after_head_only_iters():
    from models.model_plain import ModelPlain

    model = ModelPlain.__new__(ModelPlain)
    model.opt_train = {"fusion_warmup": {"head_only_iters": 2}}
    model.fix_iter = 10
    model.netG = _WarmupAwareNet()

    stage = model._configure_fusion_warmup_trainability(current_step=3)

    assert stage == "token_mixer"
    assert model.netG.fusion_adapter.operator.last_stage == "token_mixer"


def test_model_plain_optimizer_keeps_frozen_mamba_params_for_later_unfreeze():
    from models.model_plain import ModelPlain
    from models.fusion.factory import create_fusion_operator

    model = ModelPlain.__new__(ModelPlain)
    model.opt_train = {
        "G_optimizer_type": "adam",
        "G_optimizer_lr": 1e-4,
        "G_optimizer_betas": [0.9, 0.99],
        "G_optimizer_wd": 0.0,
    }
    model.netG = nn.Module()
    model.netG.fusion_operator = create_fusion_operator("mamba", 3, 1, 3, {"token_dim": 8, "token_stride": 2, "num_layers": 1})
    model.netG.fusion_operator.set_warmup_stage("writeback_only")

    model.define_optimizer()

    optimizer_param_ids = {
        id(param)
        for group in model.G_optimizer.param_groups
        for param in group["params"]
    }
    frozen_param_ids = {id(param) for param in model.netG.fusion_operator.rgb_context_encoder.parameters()}
    assert frozen_param_ids <= optimizer_param_ids
```

- [ ] **Step 2: Run the warmup/optimizer tests to verify they fail**

Run:

```bash
python -m pytest tests/models/test_two_phase_training.py -k "writeback_only_stage or token_mixer_stage or optimizer_keeps_frozen_mamba_params" -v
```

Expected: FAIL because `ModelPlain` does not yet expose warmup-stage helpers and `define_optimizer()` drops frozen params.

- [ ] **Step 3: Implement warmup-stage handoff and all-param optimizer coverage**

Update `models/model_plain.py` with these helpers and optimizer behavior:

```python
def define_optimizer(self):
    G_optim_params = []
    for name, param in self.netG.named_parameters():
        G_optim_params.append(param)
        if not param.requires_grad:
            print(f'Params [{name}] are frozen at optimizer build time but kept for later unfreeze.')
    if self.opt_train['G_optimizer_type'] == 'adam':
        self.G_optimizer = Adam(
            G_optim_params,
            lr=self.opt_train['G_optimizer_lr'],
            betas=self.opt_train['G_optimizer_betas'],
            weight_decay=self.opt_train['G_optimizer_wd'],
        )
    else:
        raise NotImplementedError


def _fusion_warmup_cfg(self) -> dict:
    return self.opt_train.get("fusion_warmup", {}) or {}


def _resolve_fusion_warmup_stage(self, current_step):
    if not self._is_phase1_step(current_step):
        return "full"
    head_only_iters = int(self._fusion_warmup_cfg().get("head_only_iters", 0))
    if current_step < head_only_iters:
        return "writeback_only"
    return "token_mixer"


def _configure_fusion_warmup_trainability(self, current_step):
    vrt = self.get_bare_model(self.netG)
    if not getattr(vrt, "fusion_enabled", False):
        return "full"
    operator = getattr(getattr(vrt, "fusion_adapter", None), "operator", None)
    if operator is None or not hasattr(operator, "set_warmup_stage"):
        return "full"
    stage = self._resolve_fusion_warmup_stage(current_step)
    operator.set_warmup_stage(stage)
    return stage
```

Then call the handoff at the start of `optimize_parameters()`:

```python
def optimize_parameters(self, current_step):
    self.timer.current_timings.clear()
    fusion_warmup_stage = self._configure_fusion_warmup_trainability(current_step)
    self.log_dict['fusion_warmup_stage'] = fusion_warmup_stage
    is_phase1 = (
        hasattr(self, 'fix_iter')
        and self.fix_iter > 0
        and current_step < self.fix_iter
    )

    with self.timer.timer('zero_grad'):
        self.G_optimizer.zero_grad()
```

- [ ] **Step 4: Run the warmup/optimizer tests to verify they pass**

Run:

```bash
python -m pytest tests/models/test_two_phase_training.py -k "writeback_only_stage or token_mixer_stage or optimizer_keeps_frozen_mamba_params" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/model_plain.py tests/models/test_two_phase_training.py
git commit -m "feat(train): add phase1 mamba warmup stage handoff"
```

### Task 3: Add a real `ModelVRT.optimize_parameters()` regression for warmup-stage switching

**Files:**
- Modify: `tests/models/test_vrt_fusion_integration.py`
- Modify: `tests/models/test_two_phase_training.py`

- [ ] **Step 1: Write the failing `ModelVRT` regression test**

Add this test to `tests/models/test_vrt_fusion_integration.py`:

```python
def test_model_vrt_optimize_parameters_switches_mamba_warmup_stage_and_phase2_unfreezes(monkeypatch):
    from collections import OrderedDict
    from contextlib import nullcontext
    from models.model_vrt import ModelVRT

    class _WarmupAwareOperator:
        def __init__(self):
            self.last_stage = None

        def set_warmup_stage(self, stage):
            self.last_stage = stage

    class _BareNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion_enabled = True
            self.fusion_cfg = {"placement": "early"}
            self.fusion_adapter = type("Adapter", (), {"operator": _WarmupAwareOperator()})()
            self.spynet_conv = nn.Linear(4, 4)
            self.pa_deform_conv = nn.Linear(4, 4)
            self.backbone_base = nn.Linear(4, 4)
            self.lora_A = nn.Parameter(torch.zeros(1))
            self.lora_B = nn.Parameter(torch.zeros(1))

    model = ModelVRT.__new__(ModelVRT)
    model.opt = {"train": {"checkpoint_save": 100}, "dist": False}
    model.opt_train = {
        "fusion_warmup": {"head_only_iters": 2},
        "G_optimizer_clipgrad": None,
        "G_regularizer_orthstep": None,
        "G_regularizer_clipstep": None,
        "E_decay": 0,
        "phase2_lora_mode": True,
        "use_lora": True,
    }
    model.fix_iter = 10
    model.fix_keys = ["spynet", "pa_deform"]
    model.fix_unflagged = False
    model.timer = type("TimerStub", (), {"current_timings": {}, "timer": staticmethod(lambda *_args, **_kwargs: nullcontext())})()
    model.log_dict = OrderedDict()
    model.grad_scaler = type(
        "ScalerStub",
        (),
        {
            "is_enabled": staticmethod(lambda: False),
            "step": staticmethod(lambda _optimizer: None),
            "update": staticmethod(lambda: None),
            "scale": staticmethod(lambda value: value),
        },
    )()
    model.G_optimizer = type("OptimizerStub", (), {"zero_grad": staticmethod(lambda: None), "step": staticmethod(lambda: None)})()
    model.fusion_debug = type(
        "DebugStub",
        (),
        {
            "should_dump_phase1_last": staticmethod(lambda *args, **kwargs: False),
            "arm": staticmethod(lambda: None),
            "disarm": staticmethod(lambda: None),
        },
    )()

    bare = _BareNet()
    bare.spynet_conv.weight.requires_grad_(False)
    bare.pa_deform_conv.weight.requires_grad_(False)
    bare.lora_A.requires_grad_(False)
    bare.lora_B.requires_grad_(False)
    bare.backbone_base.weight.requires_grad_(False)
    model.netG = bare
    model.get_bare_model = lambda net: net
    model._phase1_fusion_forward = lambda: None
    model.netG_forward = lambda: setattr(model, "E", torch.zeros(1, 1, 3, 4, 4, requires_grad=True))
    model._compute_fusion_aux_loss = lambda is_phase1, current_step=None: torch.tensor(0.0, requires_grad=True)
    model.G_lossfn_weight = 0.0
    model.G_lossfn = lambda pred, target: pred.sum() * 0.0
    model.H = torch.zeros(1, 1, 3, 4, 4)

    model.optimize_parameters(current_step=0)
    assert bare.fusion_adapter.operator.last_stage == "writeback_only"

    model.optimize_parameters(current_step=3)
    assert bare.fusion_adapter.operator.last_stage == "token_mixer"

    model.optimize_parameters(current_step=10)
    assert bare.fusion_adapter.operator.last_stage == "full"
    assert bare.spynet_conv.weight.requires_grad is True
    assert bare.pa_deform_conv.weight.requires_grad is True
    assert bare.lora_A.requires_grad is True
    assert bare.lora_B.requires_grad is True
    assert bare.backbone_base.weight.requires_grad is False
```

Also add this note to `tests/models/test_two_phase_training.py` above the `ModelPlain.__new__()` warmup tests:

```python
# These tests validate the helper logic on ModelPlain directly.
# A separate ModelVRT regression test covers the real optimize_parameters path.
```

- [ ] **Step 2: Run the `ModelVRT` regression test to verify it fails**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py::test_model_vrt_optimize_parameters_switches_mamba_warmup_stage_and_phase2_unfreezes -v
```

Expected: FAIL because the real `ModelVRT.optimize_parameters()` path does not yet pass warmup stages through `ModelPlain.optimize_parameters()`, and the `current_step == fix_iter` transition still lacks explicit regression coverage.

- [ ] **Step 3: Verify the real `ModelVRT` path passes after Task 2 changes**

No new code is required in this task if Task 2 is implemented exactly as written. Re-run the real-path regression after Task 2 lands to prove the helper-only tests and the production training path agree.

- [ ] **Step 4: Run the `ModelVRT` regression test to verify it passes**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py::test_model_vrt_optimize_parameters_switches_mamba_warmup_stage_and_phase2_unfreezes -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_vrt_fusion_integration.py tests/models/test_two_phase_training.py
git commit -m "test(train): cover mamba warmup on ModelVRT path"
```

### Task 4: Add scheduled phase-1 passthrough decay and update-penalty loss

**Files:**
- Modify: `models/model_plain.py`
- Test: `tests/models/test_model_plain_fusion_aux_loss.py`

- [ ] **Step 1: Write the failing phase-1 loss schedule tests**

Update `_make_opt()` and add these tests to `tests/models/test_model_plain_fusion_aux_loss.py`:

```python
def _make_opt(phase1_aux=1.0, phase2_aux=0.2, passthrough=0.2, fix_iter=10, fusion_warmup=None):
    return {
        'scale': 1,
        'n_channels': 3,
        'netG': {
            'net_type': 'vrt',
            'input': {'strategy': 'fusion', 'mode': 'dual', 'raw_ingress_chans': 7},
            'fusion': {
                'enable': True, 'placement': 'early', 'operator': 'gated',
                'out_chans': 3, 'operator_params': {},
            },
            'in_chans': 7, 'upscale': 1,
            'img_size': [6, 8, 8], 'window_size': [6, 8, 8],
            'depths': [2, 2, 2, 2, 2, 2, 2, 2], 'indep_reconsts': [6, 7], 'embed_dims': [16] * 8, 'num_heads': [2] * 8,
            'output_mode': 'restoration',
            'restoration_reducer': {'type': 'index', 'index': 2},
            'pa_frames': 2,
            'use_flash_attn': False,
            'optical_flow': {'module': 'spynet', 'checkpoint': None, 'params': {}},
            'deformable_groups': 4,
            'nonblind_denoising': False,
            'use_checkpoint_attn': False,
            'use_checkpoint_ffn': False,
            'no_checkpoint_attn_blocks': [],
            'no_checkpoint_ffn_blocks': [],
            'dcn_type': 'DCNv2',
            'dcn_apply_softmax': False,
            'init_type': 'default',
            'init_bn_type': 'uniform',
            'init_gain': 0.2,
        },
        'train': {
            'G_lossfn_type': 'charbonnier',
            'G_lossfn_weight': 1.0,
            'G_charbonnier_eps': 1e-6,
            'phase1_fusion_aux_loss_weight': phase1_aux,
            'phase2_fusion_aux_loss_weight': phase2_aux,
            'fusion_passthrough_loss_weight': passthrough,
            'fusion_warmup': fusion_warmup or {},
            'G_optimizer_type': 'adam',
            'G_optimizer_lr': 1e-4,
            'G_optimizer_betas': [0.9, 0.99],
            'G_optimizer_wd': 0,
            'G_optimizer_clipgrad': None,
            'G_optimizer_reuse': False,
            'G_scheduler_type': 'MultiStepLR',
            'G_scheduler_milestones': [100],
            'G_scheduler_gamma': 0.5,
            'G_regularizer_orthstep': None,
            'G_regularizer_clipstep': None,
            'G_param_strict': False,
            'E_param_strict': False,
            'E_decay': 0,
            'manual_seed': 0,
            'fix_iter': fix_iter,
            'fix_keys': [],
            'checkpoint_save': 100,
            'checkpoint_test': 100,
            'checkpoint_print': 10,
            'amp': {'enable': False},
            'freeze_backbone': False,
        },
        'path': {
            'root': '/tmp',
            'models': '/tmp',
            'pretrained_netG': None,
            'pretrained_netE': None,
            'pretrained_optimizerG': None,
        },
        'rank': 0,
        'dist': False,
        'is_train': True,
    }


def test_phase1_passthrough_weight_decays_linearly_from_warmup_cfg():
    from models.model_plain import ModelPlain

    model = ModelPlain.__new__(ModelPlain)
    model.opt_train = {
        "fusion_passthrough_loss_weight": 0.2,
        "fusion_warmup": {
            "passthrough_weight_start": 0.2,
            "passthrough_weight_end": 0.0,
        },
    }
    model.fix_iter = 10

    assert model._resolve_phase1_passthrough_weight(0) == pytest.approx(0.2)
    assert model._resolve_phase1_passthrough_weight(9) == pytest.approx(0.0)


def test_fusion_aux_loss_phase1_adds_update_penalty_from_effective_update():
    from models.model_plain import ModelPlain

    model = ModelPlain(
        _make_opt(
            phase1_aux=0.0,
            phase2_aux=0.0,
            passthrough=0.0,
            fusion_warmup={"update_penalty_weight": 0.5},
        )
    )
    model.define_loss()
    fusion_main = torch.ones(1, 6, 3, 8, 8)
    _inject_fusion_hook(model, fusion_main, spike_bins=4)
    model.H = torch.ones(1, 6, 3, 8, 8)
    model.L = torch.zeros(1, 6, 7, 8, 8)

    loss = model._compute_fusion_aux_loss(is_phase1=True, current_step=0)

    assert loss.item() == pytest.approx(0.5, rel=1e-4, abs=1e-4)


def test_phase2_fusion_aux_loss_ignores_update_penalty():
    from models.model_plain import ModelPlain

    model = ModelPlain(
        _make_opt(
            phase1_aux=0.0,
            phase2_aux=0.0,
            passthrough=0.0,
            fusion_warmup={"update_penalty_weight": 0.5},
        )
    )
    model.define_loss()
    fusion_main = torch.ones(1, 6, 3, 8, 8)
    _inject_fusion_hook(model, fusion_main, spike_bins=4)
    model.H = torch.ones(1, 6, 3, 8, 8)
    model.L = torch.zeros(1, 6, 7, 8, 8)

    loss = model._compute_fusion_aux_loss(is_phase1=False, current_step=0)

    assert loss.item() == 0.0
```

- [ ] **Step 2: Run the phase-1 loss tests to verify they fail**

Run:

```bash
python -m pytest tests/models/test_model_plain_fusion_aux_loss.py -k "passthrough_weight_decays_linearly or adds_update_penalty or ignores_update_penalty" -v
```

Expected: FAIL because `ModelPlain` does not yet accept `current_step` in `_compute_fusion_aux_loss()` and does not schedule phase-1 regularization.

- [ ] **Step 3: Implement scheduled passthrough decay and update penalty**

Extend `models/model_plain.py` with these helpers and loss changes:

```python
def _resolve_phase1_passthrough_weight(self, current_step) -> float:
    warmup_cfg = self._fusion_warmup_cfg()
    start = float(warmup_cfg.get("passthrough_weight_start", self.opt_train.get("fusion_passthrough_loss_weight", 0.0)))
    end = float(warmup_cfg.get("passthrough_weight_end", self.opt_train.get("fusion_passthrough_loss_weight", 0.0)))
    if not hasattr(self, "fix_iter") or self.fix_iter <= 1:
        return end
    clamped = min(max(int(current_step or 0), 0), self.fix_iter - 1)
    progress = clamped / float(self.fix_iter - 1)
    return start + (end - start) * progress


def _compute_fusion_aux_loss(self, is_phase1: bool, current_step=None) -> torch.Tensor:
    if is_phase1:
        aux_weight = self.opt_train.get('phase1_fusion_aux_loss_weight', 0.0)
        pass_weight = self._resolve_phase1_passthrough_weight(current_step)
        update_penalty_weight = float(self._fusion_warmup_cfg().get("update_penalty_weight", 0.0))
    else:
        aux_weight = self.opt_train.get('phase2_fusion_aux_loss_weight', 0.0)
        pass_weight = 0.0
        update_penalty_weight = 0.0
    if aux_weight == 0.0 and pass_weight == 0.0 and update_penalty_weight == 0.0:
        return torch.tensor(0.0, device=self.device)
    vrt = self.get_bare_model(self.netG)
    fusion_main = getattr(vrt, "_last_fusion_main", None)
    target_frames = self.H.size(1)
    if fusion_main is not None:
        if fusion_main.size(1) != target_frames:
            raise ValueError(
                "Fusion aux loss expected canonical main timeline "
                f"N={target_frames}, got {fusion_main.size(1)}."
            )
        fusion_center = fusion_main
    else:
        fusion_out = getattr(vrt, "_last_fusion_out", None)
        if fusion_out is None:
            return torch.tensor(0.0, device=self.device)
        S = getattr(vrt, "_last_spike_bins", 0)
        if fusion_out.size(1) == target_frames:
            fusion_center = fusion_out
        elif S > 0 and fusion_out.size(1) == target_frames * S:
            fusion_center = fusion_out[:, S // 2 :: S, :, :, :]
        else:
            raise ValueError(
                "Fusion aux loss expected fusion_out time dimension to match either "
                f"N={target_frames} or N*S={target_frames * S}, got {fusion_out.size(1)} "
                f"(S={S})."
            )
    blur_rgb = self.L[:, :, :3, :, :]
    loss = torch.tensor(0.0, device=self.device)
    if aux_weight > 0.0:
        loss = loss + aux_weight * self.G_lossfn(fusion_center, self.H)
    if pass_weight > 0.0:
        loss = loss + pass_weight * self.G_lossfn(fusion_center, blur_rgb)
    if update_penalty_weight > 0.0:
        effective_update = fusion_center - blur_rgb
        loss = loss + update_penalty_weight * effective_update.abs().mean()
    return loss
```

Update `optimize_parameters()` to pass `current_step` through both call sites:

```python
if is_phase1:
    G_loss = self._compute_fusion_aux_loss(is_phase1=True, current_step=current_step)
else:
    G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
    G_loss = G_loss + self._compute_fusion_aux_loss(is_phase1=False, current_step=current_step)
```

- [ ] **Step 4: Run the phase-1 loss tests to verify they pass**

Run:

```bash
python -m pytest tests/models/test_model_plain_fusion_aux_loss.py -k "passthrough_weight_decays_linearly or adds_update_penalty or ignores_update_penalty" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/model_plain.py tests/models/test_model_plain_fusion_aux_loss.py
git commit -m "feat(train): schedule phase1 fusion regularization"
```

### Task 5: Surface diagnostics through fusion metadata and training logs, and keep runnable debug paths alive

**Files:**
- Modify: `models/fusion/adapters/early.py`
- Modify: `models/model_plain.py`
- Modify: `tests/models/test_vrt_fusion_integration.py`
- Modify: `tests/models/test_two_phase_training.py`
- Modify: `tests/models/test_fusion_debug_dumper.py`
- Modify: `options/gopro_rgbspike_server.json`
- Modify: `options/gopro_rgbspike_server_debug.json`

- [ ] **Step 1: Write the failing diagnostics/debug visibility tests**

Add this test to `tests/models/test_vrt_fusion_integration.py`:

```python
def test_vrt_collapsed_mamba_meta_merges_operator_diagnostics(monkeypatch):
    model = VRT(
        upscale=1,
        in_chans=7,
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
        opt={
            "netG": {
                "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 7},
                "fusion": {
                    "placement": "early",
                    "operator": "mamba",
                    "out_chans": 3,
                    "operator_params": {"token_dim": 8, "token_stride": 2, "num_layers": 1},
                    "early": {"expand_to_full_t": False},
                },
                "output_mode": "restoration",
            }
        },
    )

    monkeypatch.setattr(model.fusion_adapter.operator, "forward", lambda rgb, spike: rgb)
    monkeypatch.setattr(
        model.fusion_adapter.operator,
        "diagnostics",
        lambda: {
            "token_norm": 1.0,
            "mamba_norm": 2.0,
            "delta_norm": 3.0,
            "gate_mean": 0.25,
            "effective_update_norm": 0.0,
            "warmup_stage": "token_mixer",
        },
    )

    dummy_flows = [
        torch.zeros(1, 5, 2, 8, 8),
        torch.zeros(1, 5, 2, 4, 4),
        torch.zeros(1, 5, 2, 2, 2),
        torch.zeros(1, 5, 2, 1, 1),
    ]

    monkeypatch.setattr(model, "get_flows", lambda _x, flow_spike=None: (dummy_flows, dummy_flows))
    monkeypatch.setattr(
        model,
        "get_aligned_image_2frames",
        lambda _x, _fb, _ff: [torch.zeros(1, 6, model.backbone_in_chans * 4, 8, 8)] * 2,
    )
    monkeypatch.setattr(model, "forward_features", lambda _x, *_args, **_kwargs: torch.zeros_like(_x))

    x = torch.randn(1, 6, 7, 8, 8)
    with torch.no_grad():
        _ = model(x)

    assert model._last_fusion_meta["token_norm"] == 1.0
    assert model._last_fusion_meta["mamba_norm"] == 2.0
    assert model._last_fusion_meta["gate_mean"] == 0.25
    assert model._last_fusion_meta["warmup_stage"] == "token_mixer"
```

Add these tests to `tests/models/test_two_phase_training.py` and `tests/models/test_fusion_debug_dumper.py`:

```python
def test_model_plain_current_log_includes_selected_fusion_diagnostics():
    from collections import OrderedDict
    from models.model_plain import ModelPlain

    model = ModelPlain.__new__(ModelPlain)
    model.log_dict = OrderedDict([("G_loss", 1.0)])
    model.timer = type("TimerStub", (), {"get_current_timings": staticmethod(lambda: {})})()

    model._record_fusion_diagnostics_to_log(
        {
            "gate_mean": 0.25,
            "effective_update_norm": 0.01,
            "delta_norm": 0.02,
            "warmup_stage": "token_mixer",
            "token_norm": 1.5,
        }
    )

    log = model.current_log()
    assert log["fusion_gate_mean"] == 0.25
    assert log["fusion_effective_update_norm"] == 0.01
    assert log["fusion_delta_norm"] == 0.02
    assert log["fusion_warmup_stage"] == "token_mixer"


def test_tokenized_mamba_runnable_configs_use_train_batch_debug_source():
    main_text = Path("options/gopro_rgbspike_server.json").read_text(encoding="utf-8")
    debug_text = Path("options/gopro_rgbspike_server_debug.json").read_text(encoding="utf-8")

    assert '"operator": "mamba"' in main_text
    assert '"source": "train_batch"' in main_text
    assert '"source": "train_batch"' in debug_text


def test_fusion_debug_dumper_can_dump_collapsed_mamba_train_batch_view(tmp_path):
    dumper = FusionDebugDumper(_make_opt(tmp_path))
    fusion_main = torch.rand(1, 6, 3, 8, 8)

    dumped = dumper.dump_tensor(
        fusion_main=fusion_main,
        fusion_exec=fusion_main,
        fusion_meta={"frame_contract": "collapsed", "spike_bins": 4, "warmup_stage": "token_mixer"},
        current_step=5,
        folder="GOPR0003",
        gt=torch.rand(1, 6, 3, 8, 8),
        rank=0,
        source_view="main",
    )

    assert dumped is True
```

- [ ] **Step 2: Run the diagnostics/debug tests to verify they fail**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py::test_vrt_collapsed_mamba_meta_merges_operator_diagnostics tests/models/test_two_phase_training.py::test_model_plain_current_log_includes_selected_fusion_diagnostics tests/models/test_fusion_debug_dumper.py::test_tokenized_mamba_runnable_configs_use_train_batch_debug_source tests/models/test_fusion_debug_dumper.py::test_fusion_debug_dumper_can_dump_collapsed_mamba_train_batch_view -v
```

Expected: FAIL because the adapter does not yet merge operator diagnostics into `meta`, `ModelPlain` does not yet export selected fusion diagnostics into `current_log()`, and the runnable tokenized-`mamba` configs still point at the CPU full-frame replay path instead of the train-batch debug path.

- [ ] **Step 3: Implement metadata merge, dashboard logging, and the runnable debug config switch**

In `models/fusion/adapters/early.py`, add a helper and use it in both collapsed and expanded returns:

```python
def _attach_operator_diagnostics(self, meta: dict[str, Any]) -> dict[str, Any]:
    diagnostics_getter = getattr(self.operator, "diagnostics", None)
    if not callable(diagnostics_getter):
        return meta
    return {**meta, **diagnostics_getter()}
```

Then wrap the existing `meta` assignments:

```python
"meta": self._attach_operator_diagnostics(
    self._build_meta(
        frame_contract=frame_contract,
        spike_bins=spike_steps_per_frame,
        main_steps=steps,
        exec_steps=backbone_view.size(1),
        aux_steps=None,
        main_from_exec_rule=None,
    )
),
```

In `models/model_plain.py`, add a helper that copies a stable subset of diagnostics into `self.log_dict`, then call it near the end of `optimize_parameters()` using `vrt._last_fusion_meta`:

```python
def _record_fusion_diagnostics_to_log(self, fusion_meta) -> None:
    if not isinstance(fusion_meta, dict):
        return
    for key in ("token_norm", "mamba_norm", "delta_norm", "gate_mean", "effective_update_norm"):
        if key in fusion_meta:
            self.log_dict[f"fusion_{key}"] = float(fusion_meta[key])
    if "warmup_stage" in fusion_meta:
        self.log_dict["fusion_warmup_stage"] = str(fusion_meta["warmup_stage"])
```

```python
vrt = self.get_bare_model(self.netG)
self._record_fusion_diagnostics_to_log(getattr(vrt, "_last_fusion_meta", None))
```

Update the two runnable configs so both the full run and the debug run use the new operator params, the warmup block, and a `train_batch` phase-1 debug source. For tokenized `mamba`, this is intentional: keep `phase1_last` dumps on the captured train-batch tensors instead of `val_full_frame`, because the current full-frame replay path deep-copies the adapter to CPU and is incompatible with the CUDA-only `mamba_ssm` execution path.

```json
"fusion": {
  "enable": true,
  "placement": "early",
  "operator": "mamba",
  "mode": "replace",
  "out_chans": 3,
  "operator_params": {
    "token_dim": 48,
    "token_stride": 4,
    "d_state": 32,
    "d_conv": 4,
    "expand": 2,
    "num_layers": 3,
    "alpha_init": 0.05,
    "gate_bias_init": -2.0,
    "enable_diagnostics": true
  },
  "debug": {
    "enable": true,
    "save_images": true,
    "trigger": "phase1_last",
    "source": "train_batch",
    "subdir": "fusion_phase1_last_train",
    "max_batches": 1,
    "max_frames": 24
  }
},
```

```json
"fusion_warmup": {
  "head_only_iters": 2000,
  "passthrough_weight_start": 0.2,
  "passthrough_weight_end": 0.0,
  "update_penalty_weight": 0.01
},
```

- [ ] **Step 4: Run the focused regression suite**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py tests/models/test_two_phase_training.py tests/models/test_model_plain_fusion_aux_loss.py tests/models/test_vrt_fusion_integration.py tests/models/test_fusion_debug_dumper.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion/adapters/early.py models/model_plain.py tests/models/test_vrt_fusion_integration.py tests/models/test_two_phase_training.py tests/models/test_fusion_debug_dumper.py options/gopro_rgbspike_server.json options/gopro_rgbspike_server_debug.json
git commit -m "feat(fusion): log mamba diagnostics and keep debug path runnable"
```
