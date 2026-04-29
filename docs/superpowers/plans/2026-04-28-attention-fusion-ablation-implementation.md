# Attention Fusion Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `attention` as a tokenized early-fusion operator that matches the current `mamba` structure closely enough for a fair collapsed-contract temporal-mixer ablation, while still remaining compatible with later expanded-contract experiments through the existing early-fusion adapter.

**Architecture:** Implement `AttentionFusionOperator` with the same RGB encoder, spike token encoder, write-back head, warmup staging, and diagnostics pattern as tokenized `mamba`, replacing only the temporal mixer with lightweight self-attention blocks. Update VRT so `attention` behaves like `mamba` at the contract level: default `collapsed`, explicitly expandable through `frame_contract="expanded"`, and logged through the same fusion metadata path.

**Tech Stack:** Python, PyTorch, pytest, existing VRT/fusion adapter stack

---

## File Map

- Create: `models/fusion/operators/attention.py`
  - New tokenized early-fusion operator with `_AttentionBlock`, `rgb_context_encoder`, `spike_token_encoder`, `attention_token_mixer`, `fusion_writeback_head`, `alpha`, `set_warmup_stage()`, and `diagnostics()`.
- Modify: `models/fusion/operators/__init__.py`
  - Register `attention` in `build_operator()` and `__all__`.
- Modify: `models/architectures/vrt/vrt.py`
  - Treat `attention` like `mamba` for early-fusion contract defaults.
  - Make `operator_default` resolve to `collapsed` for `attention`.
  - Allow `attention + frame_contract="expanded"` while rejecting `expand_to_full_t=true` under effective collapsed semantics.
- Modify: `models/model_plain.py`
  - Extend `_record_fusion_diagnostics_to_log()` to export `attention_norm`.
- Modify: `tests/models/test_fusion_early_adapter.py`
  - Lock operator shape, startup behavior, warmup freezing, and expanded adapter compatibility.
- Modify: `tests/models/test_vrt_fusion_integration.py`
  - Lock VRT default/expanded contract behavior and collapsed attention metadata propagation.
- Modify: `tests/models/test_two_phase_training.py`
  - Lock training-log propagation of `attention_norm`.
- Create: `options/gopro_rgbspike_server_attention.json`
  - Runnable collapsed-contract ablation config for `attention`.

### Task 1: Lock `AttentionFusionOperator` Semantics in Unit Tests

**Files:**
- Modify: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: Write the failing operator tests**

Update `tests/models/test_fusion_early_adapter.py` so the operator-frame-contract metadata parameterization becomes:

```python
@pytest.mark.parametrize(
    ("operator_name", "expected_frame_contract"),
    [
        ("gated", "expanded"),
        ("concat", "expanded"),
        ("pase", "expanded"),
        ("mamba", "collapsed"),
        ("attention", "collapsed"),
    ],
)
def test_fusion_operator_frame_contract_metadata(operator_name, expected_frame_contract):
    op = create_fusion_operator(operator_name, 3, 1, 3, {})
    assert getattr(op, "frame_contract", None) == expected_frame_contract
```

Add these new tests in the same file:

```python
def test_attention_operator_shape():
    op = create_fusion_operator(
        "attention",
        3,
        1,
        3,
        {
            "token_dim": 16,
            "token_stride": 2,
            "num_layers": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
        },
    )
    rgb_feat = torch.randn(2, 5, 3, 16, 16)
    spike_feat = torch.randn(2, 5, 6, 16, 16)
    out = op(rgb_feat, spike_feat)
    assert out.shape == (2, 5, 3, 16, 16)


def test_attention_operator_small_output_non_degenerate_init():
    op = create_fusion_operator(
        "attention",
        3,
        1,
        3,
        {
            "token_dim": 24,
            "token_stride": 2,
            "num_layers": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
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
        "token_norm",
        "attention_norm",
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


def test_attention_operator_writeback_only_stage_freezes_encoders_and_mixer():
    op = create_fusion_operator(
        "attention",
        3,
        1,
        3,
        {"token_dim": 16, "token_stride": 2, "num_layers": 1, "num_heads": 4},
    )

    op.set_warmup_stage("writeback_only")

    assert all(not p.requires_grad for p in op.rgb_context_encoder.parameters())
    assert all(not p.requires_grad for p in op.spike_token_encoder.parameters())
    assert all(not p.requires_grad for p in op.attention_token_mixer.parameters())
    assert all(p.requires_grad for p in op.fusion_writeback_head.parameters())
    assert op.alpha.requires_grad is True


def test_attention_operator_token_mixer_stage_unfreezes_all_feature_paths():
    op = create_fusion_operator(
        "attention",
        3,
        1,
        3,
        {"token_dim": 16, "token_stride": 2, "num_layers": 1, "num_heads": 4},
    )

    op.set_warmup_stage("token_mixer")

    assert all(p.requires_grad for p in op.rgb_context_encoder.parameters())
    assert all(p.requires_grad for p in op.spike_token_encoder.parameters())
    assert all(p.requires_grad for p in op.attention_token_mixer.parameters())
    assert all(p.requires_grad for p in op.fusion_writeback_head.parameters())
    assert op.alpha.requires_grad is True


def test_early_adapter_attention_expanded_override_reuses_collapsed_operator():
    op = create_fusion_operator(
        "attention",
        3,
        1,
        3,
        {"token_dim": 16, "token_stride": 2, "num_layers": 1, "num_heads": 4},
    )
    adapter = EarlyFusionAdapter(
        operator=op,
        spike_chans=4,
        frame_contract="expanded",
    )
    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)

    result = adapter(rgb=rgb, spike=spike)

    assert result["fused_main"].shape == (1, 2, 3, 8, 8)
    assert result["backbone_view"].shape == (1, 8, 3, 8, 8)
    assert result["meta"]["requested_frame_contract"] == "expanded"
    assert result["meta"]["frame_contract"] == "expanded"
```

- [ ] **Step 2: Run the focused operator tests to verify they fail**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py -k "attention" -v
```

Expected: FAIL with `Unknown fusion operator: attention`.

- [ ] **Step 3: Implement the new operator and registry plumbing**

Create `models/fusion/operators/attention.py`:

```python
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class _AttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: float, attn_drop: float, proj_drop: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                f"Attention block requires model_dim divisible by num_heads, got {model_dim} and {num_heads}."
            )
        hidden_dim = max(model_dim, int(model_dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + self.proj_drop(attn_out)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class AttentionFusionOperator(nn.Module):
    expects_structured_early = True
    frame_contract = "collapsed"

    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int, operator_params: Dict):
        super().__init__()
        if rgb_chans != 3:
            raise ValueError("AttentionFusionOperator requires rgb_chans=3.")
        if spike_chans != 1:
            raise ValueError("AttentionFusionOperator requires spike_chans=1 at construction time.")
        if out_chans != 3:
            raise ValueError("AttentionFusionOperator requires out_chans=3.")

        token_dim = int(operator_params.get("token_dim", operator_params.get("model_dim", 48)))
        token_stride = int(operator_params.get("token_stride", 4))
        num_layers = int(operator_params.get("num_layers", 3))
        num_heads = int(operator_params.get("num_heads", 4))
        mlp_ratio = float(operator_params.get("mlp_ratio", 2.0))
        attn_drop = float(operator_params.get("attn_drop", 0.0))
        proj_drop = float(operator_params.get("proj_drop", 0.0))
        alpha_init = float(operator_params.get("alpha_init", 0.05))
        gate_bias_init = float(operator_params.get("gate_bias_init", operator_params.get("init_gate_bias", -2.0)))
        enable_diagnostics = bool(operator_params.get("enable_diagnostics", False))

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
        self.attention_token_mixer = nn.ModuleList(
            [
                _AttentionBlock(
                    model_dim=token_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(num_layers)
            ]
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

    def set_warmup_stage(self, stage) -> None:
        normalized = "full" if stage in {None, "", "full"} else str(stage).strip().lower()
        if normalized not in {"full", "writeback_only", "token_mixer"}:
            raise ValueError(f"Unsupported Attention warmup stage: {stage!r}")
        self._warmup_stage = normalized
        token_trainable = normalized != "writeback_only"
        for module in (self.rgb_context_encoder, self.spike_token_encoder, self.attention_token_mixer):
            for param in module.parameters():
                param.requires_grad_(token_trainable)
        for param in self.fusion_writeback_head.parameters():
            param.requires_grad_(True)
        self.alpha.requires_grad_(True)

    def diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() != 5:
            raise ValueError("attention early fusion expects rgb with shape [B, T, 3, H, W].")
        if spike_feat.dim() != 5:
            raise ValueError("attention early fusion expects spike with shape [B, T, S, H, W].")

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

        spike_tokens = spike_low.permute(0, 1, 4, 5, 2, 3).reshape(
            bsz * steps * token_h * token_w, spike_bins, token_dim
        )
        rgb_tokens = rgb_low.permute(0, 1, 3, 4, 2).reshape(
            bsz * steps * token_h * token_w, 1, token_dim
        )
        seq = (spike_tokens + rgb_tokens).contiguous()
        for block in self.attention_token_mixer:
            seq = block(seq)

        pooled = seq.mean(dim=1).reshape(bsz, steps, token_h, token_w, token_dim).permute(0, 1, 4, 2, 3)
        fused_low = pooled + rgb_low

        writeback = self.fusion_writeback_head["body"](fused_low.reshape(bsz * steps, token_dim, token_h, token_w))
        delta_low = self.fusion_writeback_head["delta"](writeback)
        gate_logits_low = self.fusion_writeback_head["gate"](writeback)
        delta = F.interpolate(delta_low, size=(height, width), mode="bilinear", align_corners=False).reshape(
            bsz, steps, 3, height, width
        )
        gate_logits = F.interpolate(
            gate_logits_low,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(bsz, steps, 3, height, width)
        gate = torch.sigmoid(gate_logits)
        effective_update = self.alpha.view(1, 1, 3, 1, 1) * gate * delta
        out = rgb_feat + effective_update

        if self.enable_diagnostics:
            self._last_diagnostics = {
                "token_norm": float(spike_tokens.detach().float().norm(dim=-1).mean().item()),
                "attention_norm": float(seq.detach().float().norm(dim=-1).mean().item()),
                "delta_norm": float(delta.detach().float().abs().mean().item()),
                "gate_mean": float(gate.detach().float().mean().item()),
                "effective_update_norm": float(effective_update.detach().float().abs().mean().item()),
                "warmup_stage": self._warmup_stage,
            }
        else:
            self._last_diagnostics = {"warmup_stage": self._warmup_stage}
        return out


__all__ = ["AttentionFusionOperator"]
```

Update `models/fusion/operators/__init__.py`:

```python
from .attention import AttentionFusionOperator
```

```python
    if normalized_name == 'attention':
        return AttentionFusionOperator(
            rgb_chans=rgb_chans,
            spike_chans=spike_chans,
            out_chans=out_chans,
            operator_params=operator_params,
        )
```

```python
    'AttentionFusionOperator',
```

- [ ] **Step 4: Run the focused operator tests to verify they pass**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py -k "attention" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion/operators/attention.py models/fusion/operators/__init__.py tests/models/test_fusion_early_adapter.py
git commit -m "feat(fusion): add attention fusion operator"
```

### Task 2: Wire `attention` Through VRT Contract Selection

**Files:**
- Modify: `models/architectures/vrt/vrt.py`
- Modify: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Write the failing VRT contract tests**

Add these tests to `tests/models/test_vrt_fusion_integration.py`:

```python
def test_vrt_allows_attention_expanded_frame_contract_config():
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
                "operator": "attention",
                "out_chans": 3,
                "early": {
                    "frame_contract": "expanded",
                    "expand_to_full_t": True,
                },
                "operator_params": {
                    "token_dim": 8,
                    "token_stride": 2,
                    "num_layers": 1,
                    "num_heads": 2,
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

    assert model.fusion_adapter.requested_frame_contract == "expanded"
    assert model.fusion_adapter.frame_contract == "expanded"


def test_vrt_attention_operator_name_is_case_insensitive_for_default_contract():
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
                "operator": "Attention",
                "out_chans": 3,
                "operator_params": {
                    "token_dim": 8,
                    "token_stride": 2,
                    "num_layers": 1,
                    "num_heads": 2,
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

    assert model.fusion_adapter.requested_frame_contract == "operator_default"
    assert model.fusion_adapter.frame_contract == "collapsed"
```

- [ ] **Step 2: Run the focused VRT tests to verify they fail**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py -k "attention and contract" -v
```

Expected: FAIL because VRT still treats only `mamba` as a default-collapsed operator.

- [ ] **Step 3: Implement VRT default-contract and expanded-allowance wiring**

Update `models/architectures/vrt/vrt.py`:

```python
            operator_default_contract = 'collapsed' if normalized_operator_name in {'mamba', 'attention'} else 'expanded'
```

Update the early-operator validation block:

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
```

Do not change the actual early operator constructor arguments for `attention`; it should still be built with `spike_chans=1` like `mamba`.

- [ ] **Step 4: Run the focused VRT tests to verify they pass**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py -k "attention and contract" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/architectures/vrt/vrt.py tests/models/test_vrt_fusion_integration.py
git commit -m "feat(vrt): treat attention as configurable collapsed fusion"
```

### Task 3: Verify Collapsed Attention Diagnostics Reach VRT Metadata

**Files:**
- Modify: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Write the failing metadata propagation test**

Add this test to `tests/models/test_vrt_fusion_integration.py`:

```python
def test_vrt_collapsed_attention_meta_merges_operator_diagnostics(monkeypatch):
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
                    "operator": "attention",
                    "out_chans": 3,
                    "operator_params": {
                        "token_dim": 8,
                        "token_stride": 2,
                        "num_layers": 1,
                        "num_heads": 2,
                    },
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
            "attention_norm": 2.0,
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
    assert model._last_fusion_meta["attention_norm"] == 2.0
    assert model._last_fusion_meta["gate_mean"] == 0.25
    assert model._last_fusion_meta["warmup_stage"] == "token_mixer"
```

- [ ] **Step 2: Run the focused metadata propagation test to verify it fails**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py::test_vrt_collapsed_attention_meta_merges_operator_diagnostics -v
```

Expected: FAIL before the operator exists or before VRT can build with `attention`.

- [ ] **Step 3: Re-run the focused metadata propagation test after Tasks 1-2 code lands**

Run:

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py::test_vrt_collapsed_attention_meta_merges_operator_diagnostics -v
```

Expected: PASS without additional code changes, proving the generic early-adapter diagnostics merge works for `attention` too.

- [ ] **Step 4: Commit**

```bash
git add tests/models/test_vrt_fusion_integration.py
git commit -m "test(fusion): cover attention diagnostics through VRT metadata"
```

### Task 4: Export `attention_norm` to Training Logs

**Files:**
- Modify: `models/model_plain.py`
- Modify: `tests/models/test_two_phase_training.py`

- [ ] **Step 1: Write the failing training-log test**

Add this test to `tests/models/test_two_phase_training.py` after the existing fusion diagnostics log test:

```python
def test_model_plain_current_log_includes_selected_attention_diagnostics():
    from collections import OrderedDict
    from models.model_plain import ModelPlain

    model = ModelPlain.__new__(ModelPlain)
    model.log_dict = OrderedDict([("G_loss", 1.0)])
    model.timer = type("TimerStub", (), {"get_current_timings": staticmethod(lambda: {})})()

    model._record_fusion_diagnostics_to_log(
        {
            "token_norm": 1.5,
            "attention_norm": 2.25,
            "delta_norm": 0.02,
            "gate_mean": 0.25,
            "effective_update_norm": 0.01,
            "warmup_stage": "token_mixer",
        }
    )

    log = model.current_log()
    assert log["fusion_token_norm"] == 1.5
    assert log["fusion_attention_norm"] == 2.25
    assert log["fusion_gate_mean"] == 0.25
    assert log["fusion_warmup_stage"] == "token_mixer"
```

- [ ] **Step 2: Run the focused training-log test to verify it fails**

Run:

```bash
python -m pytest tests/models/test_two_phase_training.py::test_model_plain_current_log_includes_selected_attention_diagnostics -v
```

Expected: FAIL with missing `fusion_attention_norm`.

- [ ] **Step 3: Extend `_record_fusion_diagnostics_to_log()`**

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
            "attention_norm",
            "delta_norm",
            "gate_mean",
            "effective_update_norm",
        ):
            if key in fusion_meta:
                self.log_dict[f"fusion_{key}"] = float(fusion_meta[key])
```

Keep the existing `mamba` behavior unchanged; this task only extends the stable exported key set.

- [ ] **Step 4: Run the focused training-log test to verify it passes**

Run:

```bash
python -m pytest tests/models/test_two_phase_training.py::test_model_plain_current_log_includes_selected_attention_diagnostics -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/model_plain.py tests/models/test_two_phase_training.py
git commit -m "feat(train): log attention fusion diagnostics"
```

### Task 5: Add a Runnable Collapsed Attention Ablation Config

**Files:**
- Create: `options/gopro_rgbspike_server_attention.json`

- [ ] **Step 1: Create the dedicated ablation config**

Create `options/gopro_rgbspike_server_attention.json` from the current server config with these explicit changes:

```json
"task": "gopro_tfp4_scflow_attention_tokenized"
```

```json
"fusion": {
  "enable": true,
  "placement": "early",
  "operator": "attention",
  "mode": "replace",
  "out_chans": 3,
  "operator_params": {
    "token_dim": 48,
    "token_stride": 4,
    "num_layers": 3,
    "num_heads": 4,
    "mlp_ratio": 2.0,
    "attn_drop": 0.0,
    "proj_drop": 0.0,
    "alpha_init": 0.05,
    "gate_bias_init": -2.0,
    "enable_diagnostics": true
  },
  "early": {
    "frame_contract": "collapsed"
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
}
```

Leave the rest of the training setup unchanged so the primary ablation isolates the temporal mixer choice rather than the outer experiment recipe.

- [ ] **Step 2: Validate that the config parses**

Run:

```bash
python -c "from utils import utils_option as option; opt = option.parse('options/gopro_rgbspike_server_attention.json', is_train=True); print(opt['task']); print(opt['netG']['fusion']['operator']); print(opt['netG']['fusion']['early']['frame_contract'])"
```

Expected output:

```text
gopro_tfp4_scflow_attention_tokenized
attention
collapsed
```

- [ ] **Step 3: Commit**

```bash
git add options/gopro_rgbspike_server_attention.json
git commit -m "chore(options): add attention fusion ablation config"
```

### Task 6: Run the Focused Regression Slice

**Files:**
- Modify: none
- Test: `tests/models/test_fusion_early_adapter.py`
- Test: `tests/models/test_vrt_fusion_integration.py`
- Test: `tests/models/test_two_phase_training.py`

- [ ] **Step 1: Run the combined attention regression slice**

Run:

```bash
python -m pytest tests/models/test_fusion_early_adapter.py tests/models/test_vrt_fusion_integration.py tests/models/test_two_phase_training.py -k "attention" -v
```

Expected: PASS

- [ ] **Step 2: Run the server config parser check once more**

Run:

```bash
python -c "from utils import utils_option as option; opt = option.parse('options/gopro_rgbspike_server_attention.json', is_train=True); print(opt['netG']['fusion']['operator'])"
```

Expected output:

```text
attention
```

- [ ] **Step 3: Commit the verification checkpoint**

```bash
git commit --allow-empty -m "test(fusion): verify attention fusion baseline wiring"
```
