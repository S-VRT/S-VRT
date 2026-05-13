# S-VRT Temporal Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a low-coupling, config-switchable temporal fusion system to S-VRT with `early | middle | hybrid` placement and `concat | gated | pase | mamba` operators, including stage-selective middle injection and `replace | residual` mode.

**Architecture:** Separate fusion into two layers: placement adapters (early/middle/hybrid) and reusable operators (concat/gated/pase/mamba). Integrate through factory construction in VRT and optional stage hook in `Stage.forward`, so baseline behavior is preserved when `fusion.enable=false`. Keep full-T early path tied to reconstructed fixed-length spike sequence.

**Tech Stack:** Python, PyTorch, existing VRT architecture, pytest smoke/integration tests.

---

## File Structure Map

### New files

1. `models/fusion/base.py`
   - Shared interfaces and config validation helpers.
2. `models/fusion/factory.py`
   - `create_fusion_operator` and `create_fusion_adapter`.
3. `models/fusion/__init__.py`
   - Public exports.
4. `models/fusion/operators/__init__.py`
   - Operator registry map.
5. `models/fusion/operators/concat.py`
   - Concat fusion operator.
6. `models/fusion/operators/gated.py`
   - Gated fusion operator.
7. `models/fusion/operators/pase.py`
   - PASE-based operator (reuse `PixelAdaptiveSpikeEncoder` core behavior).
8. `models/fusion/operators/mamba.py`
   - Mamba operator wrapper with explicit dependency check.
9. `models/fusion/adapters/__init__.py`
   - Adapter exports.
10. `models/fusion/adapters/early.py`
    - Early full-T expansion and fusion.
11. `models/fusion/adapters/middle.py`
    - Stage injection logic (`replace | residual`).
12. `models/fusion/adapters/hybrid.py`
    - Compose early + middle.
13. `tests/models/test_fusion_factory.py`
14. `tests/models/test_fusion_early_adapter.py`
15. `tests/models/test_fusion_middle_adapter.py`
16. `tests/models/test_vrt_fusion_integration.py`

### Modified files

1. `models/architectures/vrt/vrt.py`
   - Parse fusion config, split RGB/Spike views, call fusion adapter, wire stage hooks.
2. `models/architectures/vrt/stages.py`
   - Add optional middle-fusion callback in `Stage.forward`.
3. `options/gopro_rgbspike_local_debug.json`
   - Add sample `netG.fusion` block for smoke run.


### Task 1: Scaffold Fusion Package and Factory Contracts

**Files:**
- Create: `models/fusion/base.py`
- Create: `models/fusion/factory.py`
- Create: `models/fusion/__init__.py`
- Create: `models/fusion/operators/__init__.py`
- Create: `models/fusion/adapters/__init__.py`
- Test: `tests/models/test_fusion_factory.py`

- [ ] **Step 1: Write the failing tests for factory creation and config validation**

```python
# tests/models/test_fusion_factory.py
import pytest

from models.fusion.factory import create_fusion_operator, create_fusion_adapter


def test_create_known_operator():
    op = create_fusion_operator(
        operator_name="concat",
        rgb_chans=3,
        spike_chans=8,
        out_chans=3,
        operator_params={},
    )
    assert op is not None


def test_unknown_operator_raises():
    with pytest.raises(ValueError, match="Unknown fusion operator"):
        create_fusion_operator("unknown", 3, 8, 3, {})


def test_unknown_placement_raises():
    with pytest.raises(ValueError, match="Unknown fusion placement"):
        create_fusion_adapter(
            placement="unknown",
            operator=None,
            mode="replace",
            inject_stages=[],
        )
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/models/test_fusion_factory.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'models.fusion'`

- [ ] **Step 3: Implement base/factory/registry minimal code**

```python
# models/fusion/base.py
from typing import Protocol
import torch


class FusionOperator(Protocol):
    def __call__(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        ...


def validate_mode(mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode not in {"replace", "residual"}:
        raise ValueError(f"Unsupported fusion mode: {mode}")
    return mode
```

```python
# models/fusion/factory.py
from models.fusion.operators import build_operator
from models.fusion.adapters import build_adapter


def create_fusion_operator(operator_name, rgb_chans, spike_chans, out_chans, operator_params):
    return build_operator(operator_name, rgb_chans, spike_chans, out_chans, operator_params or {})


def create_fusion_adapter(placement, operator, mode, inject_stages, **kwargs):
    return build_adapter(placement, operator, mode=mode, inject_stages=inject_stages, **kwargs)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/models/test_fusion_factory.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion/base.py models/fusion/factory.py models/fusion/__init__.py models/fusion/operators/__init__.py models/fusion/adapters/__init__.py tests/models/test_fusion_factory.py
git commit -m "feat(fusion): scaffold fusion base interfaces and factory contracts"
```


### Task 2: Implement Concat and Gated Operators (Reusable by Early/Middle)

**Files:**
- Create: `models/fusion/operators/concat.py`
- Create: `models/fusion/operators/gated.py`
- Modify: `models/fusion/operators/__init__.py`
- Test: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: Write failing operator behavior tests**

```python
# tests/models/test_fusion_early_adapter.py
import torch

from models.fusion.factory import create_fusion_operator


def test_concat_operator_shape():
    op = create_fusion_operator("concat", 3, 1, 3, {})
    rgb = torch.randn(2, 5, 3, 16, 16)
    spk = torch.randn(2, 5, 1, 16, 16)
    y = op(rgb, spk)
    assert y.shape == (2, 5, 3, 16, 16)


def test_gated_operator_shape():
    op = create_fusion_operator("gated", 3, 1, 3, {})
    rgb = torch.randn(2, 5, 3, 16, 16)
    spk = torch.randn(2, 5, 1, 16, 16)
    y = op(rgb, spk)
    assert y.shape == (2, 5, 3, 16, 16)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/models/test_fusion_early_adapter.py -v`  
Expected: FAIL with unknown operator or missing class import

- [ ] **Step 3: Implement concat and gated operators**

```python
# models/fusion/operators/concat.py
import torch
import torch.nn as nn


class ConcatFusionOperator(nn.Module):
    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int):
        super().__init__()
        self.proj = nn.Conv2d(rgb_chans + spike_chans, out_chans, kernel_size=1)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = rgb_feat.shape
        x = torch.cat([rgb_feat, spike_feat], dim=2).reshape(b * t, -1, h, w)
        y = self.proj(x).reshape(b, t, -1, h, w)
        return y
```

```python
# models/fusion/operators/gated.py
import torch
import torch.nn as nn


class GatedFusionOperator(nn.Module):
    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int, hidden_chans: int = 32):
        super().__init__()
        in_ch = rgb_chans + spike_chans
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, hidden_chans, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_chans, out_chans, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, hidden_chans, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_chans, out_chans, 1),
            nn.Sigmoid(),
        )
        self.rgb_proj = nn.Conv2d(rgb_chans, out_chans, 1)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = rgb_feat.shape
        x = torch.cat([rgb_feat, spike_feat], dim=2).reshape(b * t, -1, h, w)
        rgb = rgb_feat.reshape(b * t, rgb_feat.size(2), h, w)
        y = self.fuse(x) * self.gate(x) + self.rgb_proj(rgb)
        return y.reshape(b, t, -1, h, w)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/models/test_fusion_early_adapter.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion/operators/concat.py models/fusion/operators/gated.py models/fusion/operators/__init__.py tests/models/test_fusion_early_adapter.py
git commit -m "feat(fusion): add reusable concat and gated fusion operators"
```


### Task 3: Implement PASE and Mamba Operators with Explicit Runtime Guard

**Files:**
- Create: `models/fusion/operators/pase.py`
- Create: `models/fusion/operators/mamba.py`
- Modify: `models/fusion/operators/__init__.py`
- Test: `tests/models/test_fusion_factory.py`

- [ ] **Step 1: Add failing tests for operator availability and fallback**

```python
def test_pase_operator_constructs():
    op = create_fusion_operator("pase", 3, 8, 3, {})
    assert op is not None


def test_mamba_operator_missing_dep_raises_runtime():
    op = create_fusion_operator("mamba", 3, 8, 3, {})
    # runtime check happens on forward
    import torch
    with pytest.raises(RuntimeError, match="mamba_ssm is required"):
        op(torch.randn(1, 2, 3, 8, 8), torch.randn(1, 2, 8, 8, 8))
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/models/test_fusion_factory.py -v`  
Expected: FAIL for unsupported operator names

- [ ] **Step 3: Implement operators**

```python
# models/fusion/operators/pase.py
import torch
import torch.nn as nn
from models.spk_encoder import PixelAdaptiveSpikeEncoder


class PaseFusionOperator(nn.Module):
    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int, kernel_size: int = 3, hidden_chans: int = 32):
        super().__init__()
        self.pase = PixelAdaptiveSpikeEncoder(
            in_chans=spike_chans,
            out_chans=out_chans,
            kernel_size=kernel_size,
            hidden_chans=hidden_chans,
            normalize_kernel=True,
        )
        self.rgb_proj = nn.Conv2d(rgb_chans, out_chans, 1)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = spike_feat.shape
        spk = spike_feat.reshape(b * t, spike_feat.size(2), h, w)
        rgb = rgb_feat.reshape(b * t, rgb_feat.size(2), h, w)
        y = self.pase(spk) + self.rgb_proj(rgb)
        return y.reshape(b, t, -1, h, w)
```

```python
# models/fusion/operators/mamba.py
import torch
import torch.nn as nn


class MambaFusionOperator(nn.Module):
    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int):
        super().__init__()
        self.rgb_proj = nn.Conv2d(rgb_chans, out_chans, 1)
        self.spk_proj = nn.Conv2d(spike_chans, out_chans, 1)
        self._mamba_cls = None
        try:
            from mamba_ssm import Mamba  # type: ignore
            self._mamba_cls = Mamba
            self.mamba = Mamba(d_model=out_chans, d_state=16, d_conv=4, expand=2)
        except Exception:
            self.mamba = None

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if self.mamba is None:
            raise RuntimeError("mamba_ssm is required for fusion.operator='mamba'.")
        b, t, _, h, w = rgb_feat.shape
        x = self.rgb_proj(rgb_feat.reshape(b * t, rgb_feat.size(2), h, w)) + \
            self.spk_proj(spike_feat.reshape(b * t, spike_feat.size(2), h, w))
        x = x.reshape(b, t, -1, h, w).permute(0, 3, 4, 1, 2).reshape(b * h * w, t, -1)
        y = self.mamba(x)
        y = y.reshape(b, h, w, t, -1).permute(0, 3, 4, 1, 2)
        return y
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/models/test_fusion_factory.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion/operators/pase.py models/fusion/operators/mamba.py models/fusion/operators/__init__.py tests/models/test_fusion_factory.py
git commit -m "feat(fusion): add pase and mamba operator implementations with runtime guard"
```


### Task 4: Implement Early Adapter for Full-T Expansion (`N -> N*T`)

**Files:**
- Create: `models/fusion/adapters/early.py`
- Modify: `models/fusion/adapters/__init__.py`
- Test: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: Add failing test for time expansion**

```python
from models.fusion.adapters.early import EarlyFusionAdapter


def test_early_adapter_expands_time():
    import torch
    from models.fusion.factory import create_fusion_operator
    op = create_fusion_operator("concat", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 12, 12)
    y = adapter(rgb=rgb, spike=spike)
    assert y.shape == (2, 48, 3, 12, 12)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/models/test_fusion_early_adapter.py::test_early_adapter_expands_time -v`  
Expected: FAIL with missing adapter class

- [ ] **Step 3: Implement adapter**

```python
# models/fusion/adapters/early.py
import torch
import torch.nn as nn


class EarlyFusionAdapter(nn.Module):
    def __init__(self, operator: nn.Module):
        super().__init__()
        self.operator = operator

    def forward(self, rgb: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        # rgb: [B,N,3,H,W], spike: [B,N,T,H,W]
        b, n, _, h, w = rgb.shape
        t = spike.size(2)
        rgb_rep = rgb.unsqueeze(2).expand(b, n, t, rgb.size(2), h, w)
        rgb_rep = rgb_rep.reshape(b, n * t, rgb.size(2), h, w)
        spk = spike.reshape(b, n * t, 1, h, w) if spike.dim() == 5 else spike
        if spk.size(2) != 1:
            # Keep operator input contract generic: one spike frame channel at each expanded timestamp.
            spk = spk.mean(dim=2, keepdim=True)
        return self.operator(rgb_rep, spk)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/models/test_fusion_early_adapter.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion/adapters/early.py models/fusion/adapters/__init__.py tests/models/test_fusion_early_adapter.py
git commit -m "feat(fusion): implement early full-T fusion adapter with time expansion"
```


### Task 5: Integrate Fusion into VRT Forward Path (Early + Hybrid-Early)

**Files:**
- Modify: `models/architectures/vrt/vrt.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Add failing integration tests for parsing and early path**

```python
# tests/models/test_vrt_fusion_integration.py
import torch
from models.architectures.vrt import VRT


def test_vrt_builds_with_fusion_config():
    model = VRT(
        upscale=1,
        in_chans=11,
        img_size=[6, 16, 16],
        window_size=[2, 8, 8],
        depths=[1] * 13,
        embed_dims=[24] * 13,
        num_heads=[1] * 13,
        optical_flow={"module": "spynet"},
        pa_frames=2,
        opt={"netG": {"fusion": {"enable": True, "placement": "early", "operator": "concat", "out_chans": 3}}},
    )
    assert model is not None
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/models/test_vrt_fusion_integration.py::test_vrt_builds_with_fusion_config -v`  
Expected: FAIL with missing fusion wiring in VRT

- [ ] **Step 3: Implement VRT integration**

```python
# in VRT.__init__
fusion_cfg = ((opt or {}).get("netG", {}) or {}).get("fusion", {})
self.fusion_enabled = bool(fusion_cfg.get("enable", False))
self.fusion_cfg = fusion_cfg
if self.fusion_enabled:
    from models.fusion.factory import create_fusion_operator, create_fusion_adapter
    self.fusion_operator = create_fusion_operator(
        operator_name=fusion_cfg.get("operator", "concat"),
        rgb_chans=3,
        spike_chans=1,
        out_chans=int(fusion_cfg.get("out_chans", 3)),
        operator_params=fusion_cfg.get("operator_params", {}),
    )
    self.fusion_adapter = create_fusion_adapter(
        placement=fusion_cfg.get("placement", "early"),
        operator=self.fusion_operator,
        mode=fusion_cfg.get("mode", "replace"),
        inject_stages=fusion_cfg.get("inject_stages", []),
    )
```

```python
# in VRT.forward before flow estimation
if self.fusion_enabled and self.fusion_cfg.get("placement", "early") in {"early", "hybrid"}:
    rgb = x[:, :, :3, :, :]
    spike = x[:, :, 3:, :, :]
    x = self.fusion_adapter(rgb=rgb, spike=spike)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/models/test_vrt_fusion_integration.py::test_vrt_builds_with_fusion_config -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/architectures/vrt/vrt.py tests/models/test_vrt_fusion_integration.py
git commit -m "feat(vrt): wire configurable early/hybrid-early fusion adapter into VRT forward"
```


### Task 6: Add Middle Adapter and Stage Injection Hook (`replace` Default)

**Files:**
- Create: `models/fusion/adapters/middle.py`
- Create: `models/fusion/adapters/hybrid.py`
- Modify: `models/fusion/adapters/__init__.py`
- Modify: `models/architectures/vrt/stages.py`
- Modify: `models/architectures/vrt/vrt.py`
- Test: `tests/models/test_fusion_middle_adapter.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Add failing tests for stage-selective middle injection**

```python
# tests/models/test_fusion_middle_adapter.py
import torch
from models.fusion.factory import create_fusion_operator
from models.fusion.adapters.middle import MiddleFusionAdapter


def test_middle_replace_mode():
    op = create_fusion_operator("concat", 24, 24, 24, {})
    adapter = MiddleFusionAdapter(operator=op, mode="replace", inject_stages=[1, 3])
    x = torch.randn(1, 24, 6, 8, 8)  # [B,C,D,H,W]
    spike_ctx = torch.randn(1, 24, 6, 8, 8)
    y = adapter(stage_idx=1, x=x, spike_ctx=spike_ctx)
    assert y.shape == x.shape


def test_middle_skip_non_injected_stage():
    op = create_fusion_operator("concat", 24, 24, 24, {})
    adapter = MiddleFusionAdapter(operator=op, mode="replace", inject_stages=[2])
    x = torch.randn(1, 24, 6, 8, 8)
    spike_ctx = torch.randn(1, 24, 6, 8, 8)
    y = adapter(stage_idx=1, x=x, spike_ctx=spike_ctx)
    assert torch.equal(x, y)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/models/test_fusion_middle_adapter.py -v`  
Expected: FAIL due to missing adapter/hook

- [ ] **Step 3: Implement middle/hybrid adapters and stage hook**

```python
# models/fusion/adapters/middle.py
import torch
import torch.nn as nn
from models.fusion.base import validate_mode


class MiddleFusionAdapter(nn.Module):
    def __init__(self, operator: nn.Module, mode: str = "replace", inject_stages=None):
        super().__init__()
        self.operator = operator
        self.mode = validate_mode(mode)
        self.inject_stages = set(inject_stages or [])

    def forward(self, stage_idx: int, x: torch.Tensor, spike_ctx: torch.Tensor) -> torch.Tensor:
        if stage_idx not in self.inject_stages:
            return x
        b, c, d, h, w = x.shape
        rgb_feat = x.permute(0, 2, 1, 3, 4)
        spk_feat = spike_ctx.permute(0, 2, 1, 3, 4)
        fused = self.operator(rgb_feat, spk_feat).permute(0, 2, 1, 3, 4)
        if self.mode == "replace":
            return fused
        return x + fused
```

```python
# models/architectures/vrt/stages.py (signature change sketch)
def forward(self, x, flows_backward, flows_forward, fusion_hook=None, stage_idx=None, spike_ctx=None):
    ...
    if fusion_hook is not None and stage_idx is not None and spike_ctx is not None:
        x = fusion_hook(stage_idx=stage_idx, x=x, spike_ctx=spike_ctx)
    return x
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/models/test_fusion_middle_adapter.py tests/models/test_vrt_fusion_integration.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion/adapters/middle.py models/fusion/adapters/hybrid.py models/fusion/adapters/__init__.py models/architectures/vrt/stages.py models/architectures/vrt/vrt.py tests/models/test_fusion_middle_adapter.py tests/models/test_vrt_fusion_integration.py
git commit -m "feat(fusion): add middle/hybrid adapters with stage-selective replace/residual injection"
```


### Task 7: Configuration Samples, Guards, and End-to-End Smoke

**Files:**
- Modify: `options/gopro_rgbspike_local_debug.json`
- Modify: `tests/models/test_vrt_smoke.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Add failing test for unsupported full-T reconstruction type guard**

```python
def test_full_t_rejects_non_spikecv_tfp():
    import pytest
    from models.architectures.vrt import VRT
    with pytest.raises(ValueError, match="full-T early fusion requires spikecv_tfp"):
        VRT(
            upscale=1,
            in_chans=11,
            img_size=[6, 16, 16],
            window_size=[2, 8, 8],
            depths=[1] * 13,
            embed_dims=[24] * 13,
            num_heads=[1] * 13,
            optical_flow={"module": "spynet"},
            pa_frames=2,
            opt={"netG": {"fusion": {"enable": True, "placement": "early", "early": {"expand_to_full_t": True}},
                          "spike_reconstruction": {"type": "middle_tfp"}}},
        )
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/models/test_vrt_fusion_integration.py::test_full_t_rejects_non_spikecv_tfp -v`  
Expected: FAIL before guard exists

- [ ] **Step 3: Implement guard + sample config + smoke update**

```python
# in VRT.__init__ guard sketch
if self.fusion_enabled and self.fusion_cfg.get("placement") in {"early", "hybrid"}:
    if ((self.fusion_cfg.get("early", {}) or {}).get("expand_to_full_t", False)):
        recon_type = str((((opt or {}).get("datasets", {}) or {}).get("train", {}) or {}).get("spike_reconstruction", {}).get("type", "spikecv_tfp")
        if recon_type.lower() not in {"spikecv_tfp"}:
            raise ValueError("full-T early fusion requires spikecv_tfp reconstruction.")
```

```json
// options/gopro_rgbspike_local_debug.json
"fusion": {
  "enable": true,
  "placement": "middle",
  "operator": "gated",
  "mode": "replace",
  "inject_stages": [1, 3, 5],
  "out_chans": 11,
  "operator_params": {}
}
```

- [ ] **Step 4: Run full fusion smoke tests**

Run: `pytest tests/models/test_fusion_factory.py tests/models/test_fusion_early_adapter.py tests/models/test_fusion_middle_adapter.py tests/models/test_vrt_fusion_integration.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add options/gopro_rgbspike_local_debug.json tests/models/test_vrt_smoke.py tests/models/test_vrt_fusion_integration.py models/architectures/vrt/vrt.py
git commit -m "test(config): add fusion smoke coverage and reconstruction guard for full-T early path"
```


## Self-Review

### 1. Spec coverage check

1. Placement switch (`early/middle/hybrid`): Tasks 4, 6.
2. Operator switch (`concat/gated/pase/mamba`): Tasks 2, 3.
3. Middle stage list + mode (`replace/residual`): Task 6.
4. Full-T early based on reconstructed fixed-length spike sequence: Task 4.
5. `middle_tfp/snn` full-T guard: Task 7.
6. Low-coupling factory-based integration: Tasks 1, 5, 6.

No uncovered spec requirement found.

### 2. Placeholder scan

Searched for: `TBD`, `TODO`, `implement later`, `add validation`, `similar to`.  
None used as unresolved placeholders.

### 3. Type/interface consistency check

1. Operator interface: `forward(rgb_feat, spike_feat)` used consistently in tasks 2/3/4/6.
2. Adapter interface: early uses `forward(rgb, spike)`; middle uses `forward(stage_idx, x, spike_ctx)`; VRT/Stage integration references match task definitions.
3. Modes and placements use same literal set across tasks and config examples.

