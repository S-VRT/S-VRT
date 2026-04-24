# Early Fusion Temporal Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign EarlyFusionAdapter to support spatial upsampling (spike→RGB resolution), temporal expansion (N→N*S frames), and configurable restoration/interpolation output handling, with partial weight loading and freeze_backbone support for Stage A training.

**Architecture:** EarlyFusionAdapter gains a SpikeUpsample submodule (bilinear + 2-layer refinement conv) and rewrites forward to: (1) upsample spike spatially, (2) expand RGB×S temporally, (3) fuse via operator outputting 3 channels. VRT backbone receives N*S×3-channel frames. For `output_mode='restoration'`, a configurable reducer collapses each S-frame group back to N output frames using one of three strategies: fixed `index`, learnable `selector`, or learnable `residual_selector`. For `output_mode='interpolation'`, the full N*S sequence is kept. Partial weight loading replaces the broken positional-zip loader. `train.freeze_backbone` freezes VRT parameters for Stage A without changing pretrained backbone shapes.

**Tech Stack:** PyTorch, pytest, existing S-VRT fusion framework

**Specs:** `docs/superpowers/specs/2026-04-14-early-fusion-temporal-expansion-design.md`, `docs/superpowers/specs/2026-04-14-svrt-early-fusion-redesign.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `models/fusion/adapters/early.py` | Modify | Add `SpikeUpsample` submodule; rewrite `EarlyFusionAdapter.__init__` to accept `spike_chans`; rewrite `forward` with spatial upsample + temporal expansion |
| `models/architectures/vrt/vrt.py` | Modify | Remove old early-fusion channel-equality validation; compute `effective_in_chans` for non-SGP path; add `output_mode` + configurable restoration reducer; wire interpolation path |
| `models/fusion/reducers.py` | Create | Add `build_restoration_reducer()` and reducer modules for `index`, `selector`, and `residual_selector` |
| `models/fusion/factory.py` | Modify | Pass `spike_chans` kwarg through to adapter |
| `models/model_base.py` | Modify | Add `load_network_partial()` with key+shape matching |
| `models/model_plain.py` | Modify | Add `freeze_backbone` logic after loading; use partial loading when configured; relax `_validate_dual_input_tensors` spatial check |
| `options/gopro_rgbspike_local.json` | Modify | Add `output_mode`, `fusion`, and `reducer` config under `netG`; add Stage A switches under existing `train` block |
| `tests/models/test_fusion_early_adapter.py` | Modify | Add tests for spatial mismatch, SpikeUpsample, temporal expansion with upsample |
| `tests/models/test_vrt_fusion_integration.py` | Modify | Add tests for output_mode branching and reducer behavior |
| `tests/models/test_vrt_dual_input_priority.py` | Modify | Update/remove old tests that enforce pre-redesign early-fusion semantics |
| `tests/models/test_partial_loading.py` | Create | Tests for partial weight loading and freeze_backbone |

---

### Task 1: SpikeUpsample + EarlyFusionAdapter Rewrite

**Files:**
- Modify: `models/fusion/adapters/early.py`
- Test: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: Write failing test for SpikeUpsample**

Add to `tests/models/test_fusion_early_adapter.py`:

```python
import torch.nn.functional as F
from models.fusion.adapters.early import SpikeUpsample


def test_spike_upsample_shape():
    """SpikeUpsample should upsample spike from (6,4) to (12,12)."""
    upsample = SpikeUpsample(spike_chans=4)
    # Input: [B*N, S, spike_H, spike_W]
    spike = torch.randn(4, 4, 6, 4)
    out = upsample(spike, target_h=12, target_w=12)
    assert out.shape == (4, 4, 12, 12)


def test_spike_upsample_preserves_channels():
    """Output channels must equal input channels (S)."""
    upsample = SpikeUpsample(spike_chans=8)
    spike = torch.randn(2, 8, 5, 5)
    out = upsample(spike, target_h=10, target_w=10)
    assert out.shape == (2, 8, 10, 10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_fusion_early_adapter.py::test_spike_upsample_shape tests/models/test_fusion_early_adapter.py::test_spike_upsample_preserves_channels -v`
Expected: FAIL with `ImportError: cannot import name 'SpikeUpsample'`

- [ ] **Step 3: Implement SpikeUpsample in early.py**

Replace the full content of `models/fusion/adapters/early.py`:

```python
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeUpsample(nn.Module):
    """Upsample spike TFP frames from low resolution to RGB resolution.

    Uses bilinear interpolation for arbitrary scale factors,
    followed by 2-layer refinement conv to correct sparse interpolation artifacts.

    Args:
        spike_chans: Number of TFP bins (= S = spike_channels), treated as channel dim.
    """

    def __init__(self, spike_chans: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(spike_chans, spike_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spike_chans, spike_chans, kernel_size=3, padding=1),
        )

    def forward(self, spike: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Args:
            spike: [B_flat, S, spike_H, spike_W]  (B_flat = B*N)
            target_h, target_w: target spatial resolution (RGB resolution)
        Returns:
            [B_flat, S, target_h, target_w]
        """
        x = F.interpolate(spike, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return self.refine(x)


class EarlyFusionAdapter(nn.Module):
    def __init__(
        self,
        operator: nn.Module,
        mode: str = "replace",
        inject_stages: Optional[list] = None,
        spike_chans: int = 0,
        **kwargs: Any,
    ):
        super().__init__()
        self.operator = operator
        self.mode = mode
        self.inject_stages = inject_stages if inject_stages is not None else []
        self.kwargs = kwargs
        if spike_chans > 0:
            self.spike_upsample = SpikeUpsample(spike_chans)
        else:
            self.spike_upsample = None

    def forward(self, rgb: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb:   [B, N, 3,   rgb_H, rgb_W]
            spike: [B, N, S, spike_H, spike_W]
        Returns:
            [B, N*S, 3, rgb_H, rgb_W]
        """
        if rgb.dim() != 5:
            raise ValueError("rgb must be 5D tensor [B, N, C, H, W]")
        if spike.dim() != 5:
            raise ValueError("spike must be 5D tensor [B, N, T, H, W] or [B, N, C, H, W]")

        B, N, rgb_chans, rgb_H, rgb_W = rgb.shape
        _, _, S, spike_H, spike_W = spike.shape

        # Step 1: Spatial alignment — upsample spike to RGB resolution
        if (spike_H, spike_W) != (rgb_H, rgb_W):
            if self.spike_upsample is None:
                raise ValueError(
                    f"Spike spatial resolution ({spike_H}x{spike_W}) differs from "
                    f"RGB ({rgb_H}x{rgb_W}) but spike_chans was not set — "
                    f"cannot upsample. Pass spike_chans to EarlyFusionAdapter."
                )
            spike_flat = spike.reshape(B * N, S, spike_H, spike_W)
            spike_flat = self.spike_upsample(spike_flat, rgb_H, rgb_W)
            spike = spike_flat.reshape(B, N, S, rgb_H, rgb_W)
        else:
            # Validate batch/steps match even when spatial dims are equal
            if rgb.shape[0] != spike.shape[0] or rgb.shape[1] != spike.shape[1]:
                raise ValueError("rgb and spike must share batch and steps dimensions")

        # Step 2: Temporal expansion
        # rgb: [B, N, 3, H, W] → [B, N*S, 3, H, W]
        rgb_rep = rgb.unsqueeze(2).expand(B, N, S, rgb_chans, rgb_H, rgb_W)
        rgb_rep = rgb_rep.reshape(B, N * S, rgb_chans, rgb_H, rgb_W)
        # spike: [B, N, S, H, W] → [B, N*S, 1, H, W]
        spk = spike.reshape(B, N * S, 1, rgb_H, rgb_W)

        # Step 3: Fusion operator (output must be 3 channels)
        return self.operator(rgb_rep, spk)


__all__ = ["SpikeUpsample", "EarlyFusionAdapter"]
```

- [ ] **Step 4: Run SpikeUpsample tests to verify they pass**

Run: `pytest tests/models/test_fusion_early_adapter.py::test_spike_upsample_shape tests/models/test_fusion_early_adapter.py::test_spike_upsample_preserves_channels -v`
Expected: PASS

- [ ] **Step 5: Write failing test for spatial mismatch in EarlyFusionAdapter**

Add to `tests/models/test_fusion_early_adapter.py`:

```python
def test_early_adapter_spatial_mismatch():
    """EarlyFusionAdapter should upsample spike when spatial dims differ."""
    op = create_fusion_operator("concat", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op, spike_chans=4)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 4, 6, 4)  # different spatial resolution
    out = adapter(rgb=rgb, spike=spike)
    assert out.shape == (2, 24, 3, 12, 12)  # N*S=6*4=24


def test_early_adapter_spatial_mismatch_no_spike_chans_raises():
    """Without spike_chans, spatial mismatch should raise ValueError."""
    op = create_fusion_operator("concat", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op)  # no spike_chans
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 4, 6, 4)
    with pytest.raises(ValueError, match="cannot upsample"):
        adapter(rgb=rgb, spike=spike)
```

- [ ] **Step 6: Run spatial mismatch tests to verify they pass**

Run: `pytest tests/models/test_fusion_early_adapter.py::test_early_adapter_spatial_mismatch tests/models/test_fusion_early_adapter.py::test_early_adapter_spatial_mismatch_no_spike_chans_raises -v`
Expected: PASS

- [ ] **Step 7: Verify existing tests still pass**

Run: `pytest tests/models/test_fusion_early_adapter.py -v`
Expected: All existing tests PASS (they use same spatial resolution, so `spike_upsample=None` path is taken, and the old temporal expansion logic is preserved)

- [ ] **Step 8: Commit**

```bash
git add models/fusion/adapters/early.py tests/models/test_fusion_early_adapter.py
git commit -m "feat(fusion): add SpikeUpsample and rewrite EarlyFusionAdapter with spatial alignment + temporal expansion"
```

---

### Task 2: Factory — Pass spike_chans Through to Adapter

**Files:**
- Modify: `models/fusion/factory.py`
- Modify: `models/fusion/adapters/__init__.py`
- Modify: `models/architectures/vrt/vrt.py` (adapter creation site, lines 280-309)

The factory and adapter builder already pass `**kwargs`, so `spike_chans` flows through naturally. The change is at the VRT call site where the adapter is created.

- [ ] **Step 1: Write failing test for spike_chans passthrough**

Add to `tests/models/test_fusion_early_adapter.py`:

```python
def test_early_adapter_receives_spike_chans_via_factory():
    """Factory-created adapter should have SpikeUpsample when spike_chans > 0."""
    from models.fusion.factory import create_fusion_operator, create_fusion_adapter

    op = create_fusion_operator("gated", 3, 1, 3, {})
    adapter = create_fusion_adapter(
        placement="early",
        operator=op,
        mode="replace",
        inject_stages=[],
        spike_chans=4,
    )
    assert adapter.spike_upsample is not None
    # Verify it works with spatial mismatch
    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 4, 4)
    out = adapter(rgb=rgb, spike=spike)
    assert out.shape == (1, 8, 3, 8, 8)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/models/test_fusion_early_adapter.py::test_early_adapter_receives_spike_chans_via_factory -v`
Expected: PASS (kwargs already flow through `build_adapter` → `EarlyFusionAdapter.__init__`)

- [ ] **Step 3: Commit**

```bash
git add tests/models/test_fusion_early_adapter.py
git commit -m "test(fusion): verify spike_chans passthrough via factory"
```

---

### Task 3: VRT __init__ Fixes — Validation, conv_first, output_mode

**Files:**
- Modify: `models/architectures/vrt/vrt.py` (lines 223-227, 311-318, and new attrs)
- Test: `tests/models/test_vrt_fusion_integration.py`

Three changes in `__init__`:
1. Remove the `early_out_chans != self.in_chans` validation (line 223-227) — new design has `early_out_chans=3` but `in_chans=11`
2. Compute `effective_in_chans` for `conv_first` — when early fusion is enabled, fusion outputs 3 channels, not `in_chans`
3. Add `self.output_mode` and `self.spike_bins` attributes

- [ ] **Step 1: Write failing test for out_chans=3 with in_chans=11**

Add to `tests/models/test_vrt_fusion_integration.py`:

```python
def test_vrt_builds_with_early_fusion_out_chans_3():
    """Early fusion with out_chans=3 and in_chans=11 should NOT raise."""
    opt = {
        "netG": {
            "output_mode": "restoration",
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "gated",
                "out_chans": 3,
                "operator_params": {},
            },
        }
    }
    model = VRT(
        upscale=1,
        in_chans=11,
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
    assert model.fusion_enabled is True
    assert model.output_mode == "restoration"
    assert model.spike_bins == 8  # in_chans=11, 11-3=8
    # conv_first should accept 3*9=27 channels (not 11*9=99)
    assert model.conv_first.weight.shape[1] == 27
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_vrt_fusion_integration.py::test_vrt_builds_with_early_fusion_out_chans_3 -v`
Expected: FAIL with `ValueError: ... early out_chans (3) to match in_chans (11)`

- [ ] **Step 3: Remove line 223-227 validation**

In `models/architectures/vrt/vrt.py`, remove lines 223-227:

```python
# DELETE these lines:
            if fusion_placement in {'early', 'hybrid'} and early_out_chans != self.in_chans:
                raise ValueError(
                    f"[VRT] input_mode={self.input_mode}, placement={fusion_placement} requires "
                    f"early out_chans ({early_out_chans}) to match in_chans ({self.in_chans})."
                )
```

- [ ] **Step 4: Compute effective_in_chans for conv_first**

In `models/architectures/vrt/vrt.py`, replace the `conv_first_in_chans` computation (lines 311-318) with:

```python
        # When early fusion is enabled, it outputs `early_out_chans` channels (typically 3),
        # so conv_first must match that, not the raw `in_chans` (which includes spike channels).
        if self.fusion_enabled and fusion_placement in {'early', 'hybrid'}:
            effective_in_chans = early_out_chans
        else:
            effective_in_chans = in_chans

        if self.pa_frames:
            if self.nonblind_denoising:
                conv_first_in_chans = effective_in_chans * 9 + 1
            else:
                conv_first_in_chans = effective_in_chans * 9
        else:
            conv_first_in_chans = effective_in_chans
        self.conv_first = nn.Conv3d(conv_first_in_chans, embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))
```

- [ ] **Step 5: Add output_mode and spike_bins attributes**

Add after the `conv_first` creation (still in `__init__`):

```python
        # Output mode: 'restoration' selects N frames, 'interpolation' keeps N*S frames
        self.output_mode = opt.get('netG', {}).get('output_mode', 'restoration')
        assert self.output_mode in {'restoration', 'interpolation'}, (
            f"output_mode must be 'restoration' or 'interpolation', got '{self.output_mode}'"
        )
        # spike_bins = S = number of TFP bins per RGB frame
        # Inferred from in_chans - 3 (RGB channels) when fusion is enabled
        if self.fusion_enabled and fusion_placement in {'early', 'hybrid'}:
            self.spike_bins = in_chans - 3  # e.g., 11 - 3 = 8
        else:
            self.spike_bins = 1
```

- [ ] **Step 6: Compute spike_input_chans and pass spike_chans when creating early fusion adapter**

In `models/architectures/vrt/vrt.py`, the variable `spike_input_chans` is currently only computed inside the `if fusion_placement in {'middle', 'hybrid'}:` block (line 188). We need to compute it **before** the placement-specific branches so it's available for all placements.

Move/add the computation **before** the operator creation block (before line 256):

```python
            # Compute spike input channels for all placements that involve early fusion
            spike_input_chans = self.in_chans - 3  # e.g., 11 - 3 = 8

            operator_name = fusion_cfg.get('operator', 'concat')
            operator_params = fusion_cfg.get('operator_params', {})
```

Then in the early fusion adapter creation block (around line 266-271), add `spike_chans` kwarg:

```python
                self.fusion_adapter = create_fusion_adapter(
                    placement=fusion_placement,
                    operator=self.fusion_operator,
                    mode=fusion_mode,
                    inject_stages=inject_stages,
                    spike_chans=spike_input_chans,  # NEW: enables SpikeUpsample
                )
```

Do the same for the hybrid path (around line 302-309):

```python
                self.fusion_adapter = create_fusion_adapter(
                    placement=fusion_placement,
                    operator=early_operator,
                    mode=fusion_mode,
                    inject_stages=inject_stages,
                    early_operator=early_operator,
                    middle_operator=middle_operator,
                    spike_chans=spike_input_chans,  # NEW
                )
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/models/test_vrt_fusion_integration.py::test_vrt_builds_with_early_fusion_out_chans_3 -v`
Expected: PASS

- [ ] **Step 8: Verify core VRT fusion tests still pass, excluding legacy semantic tests updated later**

Run: `pytest tests/models/test_vrt_fusion_integration.py -v`
Expected: Core VRT fusion tests PASS. Legacy semantic tests that still enforce pre-redesign early/hybrid contracts are updated in Task 8.

- [ ] **Step 9: Commit**

```bash
git add models/architectures/vrt/vrt.py tests/models/test_vrt_fusion_integration.py
git commit -m "fix(vrt): remove early_out_chans validation, compute effective_in_chans for conv_first, add output_mode"
```

---

### Task 4: Restoration Reducer — Configurable `index`, `selector`, `residual_selector`

**Files:**
- Create: `models/fusion/reducers.py`
- Modify: `models/architectures/vrt/vrt.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

For `output_mode='restoration'`, replace hard-coded frame indexing with a configurable reducer that maps `[B, N*S, C, H, W]` to `[B, N, C, H, W]` using one of three strategies:
1. `index` — fixed configured index per S-frame group, zero extra params
2. `selector` — learnable scorer that outputs S weights per group, small extra params only in reducer
3. `residual_selector` — learnable scorer plus residual-only fusion against `x_lq_rgb`, small extra params only in reducer

Backbone compatibility requirement: these reducers are appended after the pretrained VRT body; they must not change pretrained VRT layer shapes.
`selector` and `residual_selector` introduce new reducer parameters, but those parameters are isolated from the pretrained VRT backbone and should load as newly initialized modules during partial loading.

- [ ] **Step 1: Write failing reducer tests**

Add to `tests/models/test_vrt_fusion_integration.py`:

```python
import torch

from models.fusion.reducers import build_restoration_reducer


def test_index_reducer_selects_configured_index():
    reducer = build_restoration_reducer({"type": "index", "index": 1})
    x = torch.arange(24.0).reshape(1, 6, 1, 2, 2)  # N=2, S=3
    out = reducer(x=x, spike_bins=3, base_rgb=None)
    assert out.shape == (1, 2, 1, 2, 2)
    assert torch.equal(out[:, 0], x[:, 1])
    assert torch.equal(out[:, 1], x[:, 4])


def test_selector_reducer_restores_n_frames():
    reducer = build_restoration_reducer({"type": "selector", "selector_hidden": 8})
    x = torch.randn(2, 12, 3, 8, 8)  # N=3, S=4
    out = reducer(x=x, spike_bins=4, base_rgb=None)
    assert out.shape == (2, 3, 3, 8, 8)


def test_residual_selector_reducer_uses_base_rgb_shape():
    reducer = build_restoration_reducer({"type": "residual_selector", "selector_hidden": 8})
    x = torch.randn(2, 12, 3, 8, 8)
    base = torch.randn(2, 3, 3, 8, 8)
    out = reducer(x=x, spike_bins=4, base_rgb=base)
    assert out.shape == base.shape
```

- [ ] **Step 2: Run reducer tests to verify they fail**

Run: `pytest tests/models/test_vrt_fusion_integration.py::test_index_reducer_selects_configured_index tests/models/test_vrt_fusion_integration.py::test_selector_reducer_restores_n_frames tests/models/test_vrt_fusion_integration.py::test_residual_selector_reducer_uses_base_rgb_shape -v`
Expected: FAIL with missing reducer module/factory

- [ ] **Step 3: Implement reducer modules in `models/fusion/reducers.py`**

Create `models/fusion/reducers.py`:

```python
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


class FixedIndexReducer(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = int(index)

    def forward(self, x: torch.Tensor, spike_bins: int, base_rgb: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, ns, c, h, w = x.shape
        n = ns // spike_bins
        groups = x.reshape(b, n, spike_bins, c, h, w)
        idx = max(0, min(self.index, spike_bins - 1))
        return groups[:, :, idx]


class SelectorReducer(nn.Module):
    def __init__(self, hidden: int = 32, temperature: float = 0.25, hard_infer: bool = True):
        super().__init__()
        self.temperature = float(temperature)
        self.hard_infer = bool(hard_infer)
        self.score = nn.Sequential(
            nn.Conv3d(3, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, spike_bins: int, base_rgb: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, ns, c, h, w = x.shape
        n = ns // spike_bins
        groups = x.reshape(b, n, spike_bins, c, h, w)
        score_in = groups.permute(0, 1, 3, 2, 4, 5).reshape(b * n, c, spike_bins, h, w)
        logits = self.score(score_in).mean(dim=(-1, -2)).squeeze(1) / self.temperature
        weights = torch.softmax(logits, dim=1)
        if (not self.training) and self.hard_infer:
            hard_idx = weights.argmax(dim=1)
            weights = F.one_hot(hard_idx, num_classes=spike_bins).float()
        weights = weights.view(b, n, spike_bins, 1, 1, 1)
        return (groups * weights).sum(dim=2)


class ResidualSelectorReducer(SelectorReducer):
    def forward(self, x: torch.Tensor, spike_bins: int, base_rgb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if base_rgb is None:
            raise ValueError("residual_selector requires base_rgb.")
        residual = super().forward(x=x, spike_bins=spike_bins, base_rgb=None)
        return residual


def build_restoration_reducer(cfg: Optional[Dict]) -> nn.Module:
    cfg = cfg or {"type": "index", "index": 0}
    reducer_type = str(cfg.get("type", "index")).lower()
    if reducer_type == "index":
        return FixedIndexReducer(index=cfg.get("index", 0))
    if reducer_type == "selector":
        return SelectorReducer(
            hidden=int(cfg.get("selector_hidden", 32)),
            temperature=float(cfg.get("selector_temperature", 0.25)),
            hard_infer=bool(cfg.get("selector_hard_infer", True)),
        )
    if reducer_type == "residual_selector":
        return ResidualSelectorReducer(
            hidden=int(cfg.get("selector_hidden", 32)),
            temperature=float(cfg.get("selector_temperature", 0.25)),
            hard_infer=bool(cfg.get("selector_hard_infer", True)),
        )
    raise ValueError(f"Unsupported restoration reducer type: {reducer_type}")
```

- [ ] **Step 4: Run reducer tests to verify they pass**

Run: `pytest tests/models/test_vrt_fusion_integration.py::test_index_reducer_selects_configured_index tests/models/test_vrt_fusion_integration.py::test_selector_reducer_restores_n_frames tests/models/test_vrt_fusion_integration.py::test_residual_selector_reducer_uses_base_rgb_shape -v`
Expected: PASS

- [ ] **Step 5: Wire reducer config into VRT**

In `models/architectures/vrt/vrt.py`, add in `__init__` after `self.output_mode`:

```python
        reducer_cfg = ((opt or {}).get("netG", {}) or {}).get("restoration_reducer", {})
        if self.output_mode == "restoration":
            from models.fusion.reducers import build_restoration_reducer
            self.restoration_reducer = build_restoration_reducer(reducer_cfg)
        else:
            self.restoration_reducer = None
```

In `forward`, keep `S = spike.shape[2]` recording, but replace hard-coded `S//2` frame selection with reducer invocation:

```python
                if self.output_mode == "restoration" and S > 1:
                    if self.restoration_reducer is None:
                        raise RuntimeError("restoration_reducer must be initialized for restoration mode.")
                    reduced = self.restoration_reducer(x=x, spike_bins=S, base_rgb=x_lq_rgb)
                    return reduced + x_lq_rgb
```

Reducer contract for restoration mode:
- `index` and `selector` return a residual tensor shaped `[B, N, 3, H, W]`
- `residual_selector` also returns a residual tensor shaped `[B, N, 3, H, W]`; it may use `base_rgb` internally as conditioning, but must not apply the final RGB skip connection itself

This keeps VRT's restoration head semantics unchanged: the outer return site remains the single place that applies `+ x_lq_rgb`.

Keep interpolation path as explicit RGB residual expansion:

```python
                if self.output_mode == "interpolation" and S > 1:
                    bsz, n_orig = x_lq_rgb.shape[:2]
                    chans, height, width = x_lq_rgb.shape[2:]
                    x_lq_rgb_exp = (
                        x_lq_rgb.unsqueeze(2)
                        .expand(bsz, n_orig, S, chans, height, width)
                        .reshape(bsz, n_orig * S, chans, height, width)
                    )
                    return x + x_lq_rgb_exp
```

- [ ] **Step 6: Add VRT-level tests for reducer config**

Add to `tests/models/test_vrt_fusion_integration.py`:

```python
def test_vrt_builds_with_selector_reducer():
    opt = {
        "netG": {
            "output_mode": "restoration",
            "restoration_reducer": {"type": "selector", "selector_hidden": 8},
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "gated",
                "out_chans": 3,
                "operator_params": {},
            },
        }
    }
    model = VRT(
        upscale=1,
        in_chans=11,
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
    assert model.restoration_reducer is not None
```

- [ ] **Step 7: Add interpolation-path test**

Add to `tests/models/test_vrt_fusion_integration.py`:

```python
def test_vrt_forward_interpolation_mode_keeps_all_frames(monkeypatch):
    opt = {
        "netG": {
            "output_mode": "interpolation",
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "concat",
                "out_chans": 3,
                "operator_params": {},
            },
        }
    }
    model = VRT(
        upscale=1,
        in_chans=11,
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

    dummy_flows = [
        torch.zeros(1, 1, 2, 8, 8),
        torch.zeros(1, 1, 2, 4, 4),
        torch.zeros(1, 1, 2, 2, 2),
        torch.zeros(1, 1, 2, 1, 1),
    ]

    def _fake_get_flows(_x, flow_spike=None):
        return dummy_flows, dummy_flows

    def _fake_aligned(_x, _fb, _ff):
        bsz, steps, chans, height, width = _x.shape
        return [
            torch.zeros(bsz, steps, chans * 4, height, width),
            torch.zeros(bsz, steps, chans * 4, height, width),
        ]

    def _fake_forward_features(_x, _fb, _ff, fusion_hook=None, spike_ctx=None):
        return torch.zeros_like(_x)

    monkeypatch.setattr(model, "get_flows", _fake_get_flows)
    monkeypatch.setattr(model, "get_aligned_image_2frames", _fake_aligned)
    monkeypatch.setattr(model, "forward_features", _fake_forward_features)

    x = torch.randn(1, 2, 11, 8, 8)
    out = model(x)
    assert out.shape == (1, 16, 3, 8, 8)
```

- [ ] **Step 8: Run VRT reducer tests**

Run: `pytest tests/models/test_vrt_fusion_integration.py -v`
Expected: PASS, including new reducer tests

- [ ] **Step 9: Commit**

```bash
git add models/fusion/reducers.py models/architectures/vrt/vrt.py tests/models/test_vrt_fusion_integration.py
git commit -m "feat(fusion): add configurable restoration reducers for early fusion"
```

---

### Task 5: Partial Weight Loading + freeze_backbone

**Files:**
- Modify: `models/model_base.py` (add `load_network_partial` method, ~lines 200-218)
- Modify: `models/model_plain.py` (add freeze_backbone logic, modify `load` and `init_train`)
- Create: `tests/models/test_partial_loading.py`

The current `load_network` with `strict=False` does positional-zip loading (pairs old keys with new keys by position) — this is broken for partial loading because key order may differ between checkpoints. We need a new method that matches by key name + shape.

- [ ] **Step 1: Write failing test for partial loading**

Create `tests/models/test_partial_loading.py`:

```python
import pytest
import torch
import torch.nn as nn
import os
import tempfile

from models.model_base import ModelBase


class DummyModel(nn.Module):
    def __init__(self, in_c=3, out_c=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


def _save_checkpoint(model, path):
    torch.save({"params": model.state_dict()}, path)


def test_partial_loading_matches_keys_and_shapes():
    """Partial loading should load matching keys and skip mismatched shapes."""
    # Source model: conv1(3→8), conv2(8→8)
    source = DummyModel(in_c=3, out_c=8)
    # Target model: conv1(5→8) — conv1 weight shape differs, conv2 matches
    target = DummyModel(in_c=5, out_c=8)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        ckpt_path = f.name
    try:
        _save_checkpoint(source, ckpt_path)

        # Save original target conv2 weight to verify it gets overwritten
        target_conv2_before = target.conv2.weight.clone()

        base = ModelBase.__new__(ModelBase)
        loaded, skipped = base.load_network_partial(ckpt_path, target, param_key="params")

        # conv1.weight shape differs (3 vs 5 input channels) → skipped
        assert "conv1.weight" in skipped
        # conv1.bias shape matches (both 8) → loaded
        assert "conv1.bias" in loaded
        # conv2 layers match exactly → loaded
        assert "conv2.weight" in loaded
        assert "conv2.bias" in loaded

        # Verify conv2 weight was actually copied from source
        assert torch.equal(target.conv2.weight, source.conv2.weight)
    finally:
        os.unlink(ckpt_path)


def test_partial_loading_handles_extra_keys():
    """Keys in checkpoint but not in model should be skipped."""
    source = DummyModel(in_c=3, out_c=8)
    # Add an extra parameter to source
    source.extra = nn.Linear(10, 10)

    target = DummyModel(in_c=3, out_c=8)  # no extra layer

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        ckpt_path = f.name
    try:
        _save_checkpoint(source, ckpt_path)
        base = ModelBase.__new__(ModelBase)
        loaded, skipped = base.load_network_partial(ckpt_path, target, param_key="params")

        assert "extra.weight" in skipped
        assert "extra.bias" in skipped
        assert "conv1.weight" in loaded
    finally:
        os.unlink(ckpt_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_partial_loading.py -v`
Expected: FAIL with `AttributeError: ... 'load_network_partial'`

- [ ] **Step 3: Implement load_network_partial in model_base.py**

Add after the existing `load_network` method (after line 218) in `models/model_base.py`:

```python
    def load_network_partial(self, load_path, network, param_key='params'):
        """Load pretrained weights with key+shape matching.

        Keys that exist in both checkpoint and model with matching shapes are loaded.
        Keys with shape mismatches or missing keys are skipped.

        Returns:
            (loaded_keys, skipped_keys): Lists of key names.
        """
        if hasattr(network, 'module'):
            network = network.module

        state_dict = torch.load(load_path, map_location='cpu')
        if param_key in state_dict:
            state_dict = state_dict[param_key]

        model_state = network.state_dict()
        loaded, skipped = [], []

        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded.append(k)
            else:
                skipped.append(k)

        network.load_state_dict(model_state, strict=True)
        print(f'[Partial Load] loaded {len(loaded)} keys, skipped {len(skipped)} keys.')
        if skipped:
            print(f'[Partial Load] skipped keys: {skipped[:20]}{"..." if len(skipped) > 20 else ""}')
        return loaded, skipped
```

- [ ] **Step 4: Run partial loading tests to verify they pass**

Run: `pytest tests/models/test_partial_loading.py -v`
Expected: PASS

- [ ] **Step 5: Write failing test for freeze_backbone**

Add to `tests/models/test_partial_loading.py`:

```python
def test_freeze_backbone_freezes_non_fusion_params():
    """freeze_backbone should freeze all params except fusion adapter."""
    from models.architectures.vrt.vrt import VRT

    opt = {
        "netG": {
            "output_mode": "restoration",
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "gated",
                "out_chans": 3,
                "operator_params": {},
            },
        },
        "train": {
            "freeze_backbone": True,
        },
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

    # Apply freeze
    freeze_backbone(model)

    # Backbone params should be frozen
    for name, param in model.named_parameters():
        if "fusion_adapter" in name or "fusion_operator" in name:
            assert param.requires_grad, f"Fusion param {name} should be trainable"
        else:
            assert not param.requires_grad, f"Backbone param {name} should be frozen"
```

- [ ] **Step 6: Implement freeze_backbone utility**

Add to `models/model_plain.py` as a module-level function (before the class definition):

```python
def freeze_backbone(model):
    """Freeze all parameters except fusion adapter and fusion operator.

    Used for Stage A training: only EarlyFusionAdapter (SpikeUpsample + operator)
    participates in gradient updates.
    """
    for name, param in model.named_parameters():
        if "fusion_adapter" in name or "fusion_operator" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
```

Update the test import:

```python
from models.model_plain import freeze_backbone
```

- [ ] **Step 7: Wire freeze_backbone into ModelPlain.init_train**

In `models/model_plain.py`, in `init_train()` method, add after `self.load()` (line 129):

```python
    def init_train(self):
        self.load()                           # load model

        # Stage A: freeze backbone, only train fusion adapter
        if self.opt.get('train', {}).get('freeze_backbone', False):
            bare_model = self.get_bare_model(self.netG)
            freeze_backbone(bare_model)
            frozen_count = sum(1 for p in bare_model.parameters() if not p.requires_grad)
            trainable_count = sum(1 for p in bare_model.parameters() if p.requires_grad)
            print(f'[Stage A] Frozen {frozen_count} params, trainable {trainable_count} params')

        self.netG.train()                     # set training mode, for BN
        # ... rest of init_train unchanged
```

- [ ] **Step 8: Use partial loading in ModelPlain.load when configured**

In `models/model_plain.py`, modify the `load()` method to use partial loading when `train.partial_load` is true:

```python
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            use_partial = self.opt.get('train', {}).get('partial_load', False)
            if use_partial:
                self.load_network_partial(load_path_G, self.netG, param_key='params')
            else:
                self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        # ... rest of load unchanged (netE loading)
```

- [ ] **Step 9: Run freeze_backbone test to verify it passes**

Run: `pytest tests/models/test_partial_loading.py::test_freeze_backbone_freezes_non_fusion_params -v`
Expected: PASS

- [ ] **Step 10: Run all tests**

Run: `pytest tests/models/test_partial_loading.py -v`
Expected: All PASS

- [ ] **Step 11: Commit**

```bash
git add models/model_base.py models/model_plain.py tests/models/test_partial_loading.py
git commit -m "feat(training): add partial weight loading and freeze_backbone for Stage A"
```

---

### Task 6: Config Updates

**Files:**
- Modify: `options/gopro_rgbspike_local.json`

Add `output_mode`, `fusion`, and reducer config to `netG`, and add Stage A switches under the existing top-level `train` block.

- [ ] **Step 1: Add output_mode and reducer config to `netG`**

In `options/gopro_rgbspike_local.json`, add `output_mode`, `restoration_reducer`, and `fusion` inside the `"netG"` block. The baseline restoration config should use reducer type `"index"` with an explicit `index` value from config, not an implicit code-side `S//2`.

- [ ] **Step 2: Add Stage A switches to existing `train` block**

In `options/gopro_rgbspike_local.json`, add the new keys inside the existing top-level `"train"` block:

```json
"freeze_backbone": true,
"partial_load": true,
"use_lora": false,
"lora_rank": 8,
"lora_alpha": 16,
"lora_target_modules": ["qkv", "proj"]
```

- [ ] **Step 3: Verify reducer defaults and Stage A flags are internally consistent**

Review the `"in_chans"` — it should remain `11` (3 RGB + 8 spike channels), since early fusion internally produces 3-channel fused frames before `conv_first`. For the initial baseline run:
- `restoration_reducer.type` should be `"index"`
- `restoration_reducer.index` should be set explicitly by config
- `train.freeze_backbone` should be `true`
- `train.partial_load` should be `true`

- [ ] **Step 4: Commit**

```bash
git add options/gopro_rgbspike_local.json
git commit -m "config: add output_mode and train flags for Stage A early fusion"
```

---

### Task 7: Fix Dual Input Spatial Validation And Model Ingress

**Files:**
- Modify: `models/model_plain.py`
- Modify: `tests/models/test_partial_loading.py`

The current dual-input ingress has two independent blockers for spatially mismatched RGB and spike tensors:
1. `_validate_dual_input_tensors()` rejects mismatched `H/W`
2. `_build_model_input_tensor()` still does a raw `torch.cat([L_rgb, L_spike], dim=2)`, which will crash even if validation is relaxed

This task must fix both. The minimal ingress-compatible design is: when `input_mode='dual'` and spatial sizes differ, resize `L_spike` to RGB spatial size with bilinear interpolation before concatenation. `SpikeUpsample` still remains useful inside `EarlyFusionAdapter` for learned refinement when inputs are routed separately in tests or future refactors, but the current model ingress must become cat-safe.

- [ ] **Step 1: Write failing test for dual input with spatial mismatch**

Add to `tests/models/test_partial_loading.py`:

```python
def test_build_model_input_tensor_resizes_spike_before_concat():
    """Dual input ingress should resize spike to RGB spatial size before concat."""
    from models.model_plain import ModelPlain

    mp = ModelPlain.__new__(ModelPlain)
    mp.opt = {"netG": {"input_mode": "dual", "in_chans": 7}}
    mp._mark_net_input_path = lambda marker: None
    l_rgb = torch.randn(1, 2, 3, 16, 16)
    l_spike = torch.randn(1, 2, 4, 8, 8)
    out = mp._build_model_input_tensor({"L_rgb": l_rgb, "L_spike": l_spike})
    assert out.shape == (1, 2, 7, 16, 16)


def test_validate_dual_input_allows_spatial_mismatch_but_still_checks_bt():
    """Validation should allow H/W mismatch but still reject mismatched batch/time."""
    from models.model_plain import ModelPlain
    mp = ModelPlain.__new__(ModelPlain)
    mp.opt = {"netG": {"input_mode": "dual", "in_chans": 7}}
    l_rgb = torch.randn(1, 2, 3, 16, 16)
    l_spike = torch.randn(1, 2, 4, 8, 8)
    mp._validate_dual_input_tensors(l_rgb, l_spike)
    l_spike_bad = torch.randn(1, 3, 4, 8, 8)  # T=3 vs T=2
    with pytest.raises(ValueError, match="matching"):
        mp._validate_dual_input_tensors(l_rgb, l_spike_bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_partial_loading.py::test_build_model_input_tensor_resizes_spike_before_concat tests/models/test_partial_loading.py::test_validate_dual_input_allows_spatial_mismatch_but_still_checks_bt -v`
Expected: FAIL because validation rejects spatial mismatch and raw concat cannot handle different `H/W`

- [ ] **Step 3: Relax spatial validation and resize spike in `_build_model_input_tensor`**

In `models/model_plain.py`, modify `_validate_dual_input_tensors` to only validate batch and temporal dimensions:

```python
    def _validate_dual_input_tensors(self, l_rgb, l_spike):
        if l_rgb.ndim != 5 or l_spike.ndim != 5:
            raise ValueError(
                "input_mode=dual expects L_rgb and L_spike shaped [B,T,C,H,W]. "
                f"Got L_rgb ndim={l_rgb.ndim}, L_spike ndim={l_spike.ndim}."
            )
        if l_rgb.size(2) != 3:
            raise ValueError(
                f"input_mode=dual expects L_rgb channels=3, got {l_rgb.size(2)}."
            )
        if (
            l_rgb.size(0) != l_spike.size(0)
            or l_rgb.size(1) != l_spike.size(1)
        ):
            raise ValueError(
                f"input_mode=dual requires matching [B,T] between L_rgb {tuple(l_rgb.shape)} "
                f"and L_spike {tuple(l_spike.shape)}."
            )
        # Note: spatial dims (H, W) may differ. In the current concat ingress,
        # _build_model_input_tensor() resizes spike to RGB resolution before cat.
```

Also update `_build_model_input_tensor()` so dual-input concat is safe when spatial sizes differ:

```python
import torch.nn.functional as F

    def _build_model_input_tensor(self, data):
        mode = self._resolve_input_mode()
        self._mark_net_input_path('concat_path' if mode == 'concat' else 'dual_path')
        if mode == 'concat':
            if 'L' not in data:
                raise KeyError("input_mode=concat requires data['L'].")
            return data['L']

        has_rgb = 'L_rgb' in data
        has_spike = 'L_spike' in data
        has_dual = has_rgb and has_spike
        if has_dual:
            l_rgb = data['L_rgb']
            l_spike = data['L_spike']
            self._validate_dual_input_tensors(l_rgb, l_spike)
            if l_rgb.shape[-2:] != l_spike.shape[-2:]:
                b, t, c_spike, _, _ = l_spike.shape
                target_h, target_w = l_rgb.shape[-2:]
                l_spike = F.interpolate(
                    l_spike.reshape(b * t, c_spike, l_spike.size(-2), l_spike.size(-1)),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                ).reshape(b, t, c_spike, target_h, target_w)
            return torch.cat([l_rgb, l_spike], dim=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_partial_loading.py::test_build_model_input_tensor_resizes_spike_before_concat tests/models/test_partial_loading.py::test_validate_dual_input_allows_spatial_mismatch_but_still_checks_bt -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/model_plain.py tests/models/test_partial_loading.py
git commit -m "fix(model_plain): resize spike before dual-input concat and relax spatial validation"
```

---

### Task 8: Update Old Semantic Tests

**Files:**
- Modify: `tests/models/test_vrt_dual_input_priority.py`
- Modify: `tests/models/test_vrt_fusion_integration.py`

The redesign intentionally removes the old constraint that early/hybrid fusion `out_chans` must equal `in_chans`. Update or delete tests that still enforce that legacy contract, and replace them with tests that validate the new contract: `in_chans = 3 + spike_bins`, `fusion.out_chans = 3`, and successful construction in restoration mode.

- [ ] **Step 1: Replace mismatch-raises tests in `test_vrt_dual_input_priority.py`**

Remove tests that expect early/hybrid construction to fail when `out_chans != in_chans`, and replace them with:

```python
def test_vrt_dual_early_out_chans_3_is_allowed():
    opt = {
        "netG": {
            "input_mode": "dual",
            "output_mode": "restoration",
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "concat",
                "out_chans": 3,
                "operator_params": {},
            },
        }
    }
    model = _build_vrt(opt=opt, in_chans=11)
    assert model is not None


def test_vrt_dual_hybrid_out_chans_3_is_allowed():
    opt = {
        "netG": {
            "input_mode": "dual",
            "output_mode": "restoration",
            "fusion": {
                "enable": True,
                "placement": "hybrid",
                "operator": "concat",
                "out_chans": 3,
                "middle": {"out_chans": 16},
                "inject_stages": [1],
                "operator_params": {},
            },
        }
    }
    model = _build_vrt(opt=opt, in_chans=11)
    assert model is not None
```

- [ ] **Step 2: Run the updated semantic tests**

Run: `pytest tests/models/test_vrt_dual_input_priority.py -v`
Expected: PASS, with no remaining assertions that require early/hybrid `out_chans == in_chans`

- [ ] **Step 3: Commit**

```bash
git add tests/models/test_vrt_dual_input_priority.py tests/models/test_vrt_fusion_integration.py
git commit -m "test(vrt): update legacy early fusion semantics to 3-channel restoration contract"
```

---

### Task 9: End-to-End Integration Tests

**Files:**
- Modify: `tests/models/test_vrt_fusion_integration.py`
- Modify: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: Write end-to-end test with spatial mismatch + gated operator**

Add to `tests/models/test_fusion_early_adapter.py`:

```python
def test_early_adapter_gated_spatial_mismatch():
    """Gated operator with spatial mismatch should produce correct output shape."""
    op = create_fusion_operator("gated", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op, spike_chans=8)
    rgb = torch.randn(1, 4, 3, 16, 16)
    spike = torch.randn(1, 4, 8, 8, 8)   # half resolution
    out = adapter(rgb=rgb, spike=spike)
    assert out.shape == (1, 32, 3, 16, 16)  # N*S=4*8=32


def test_early_adapter_output_is_3_channels():
    """Regardless of spike_chans, output must always be 3 channels."""
    for S in [2, 4, 8]:
        op = create_fusion_operator("gated", 3, 1, 3, {})
        adapter = EarlyFusionAdapter(operator=op, spike_chans=S)
        rgb = torch.randn(1, 2, 3, 8, 8)
        spike = torch.randn(1, 2, S, 4, 4)
        out = adapter(rgb=rgb, spike=spike)
        assert out.shape[2] == 3, f"Expected 3 channels, got {out.shape[2]} for S={S}"
        assert out.shape[1] == 2 * S
```

- [ ] **Step 2: Run and verify they pass**

Run: `pytest tests/models/test_fusion_early_adapter.py::test_early_adapter_gated_spatial_mismatch tests/models/test_fusion_early_adapter.py::test_early_adapter_output_is_3_channels -v`
Expected: PASS

- [ ] **Step 3: Write test for `index` reducer correctness**

Add to `tests/models/test_vrt_fusion_integration.py`:

```python
def test_index_reducer_uses_configured_position():
    reducer = build_restoration_reducer({"type": "index", "index": 2})
    x = torch.arange(12.0).reshape(1, 12, 1, 1, 1).expand(1, 12, 3, 8, 8)  # N=3, S=4
    selected = reducer(x=x, spike_bins=4, base_rgb=None)
    assert selected.shape == (1, 3, 3, 8, 8)
    assert selected[0, 0, 0, 0, 0].item() == 2.0
    assert selected[0, 1, 0, 0, 0].item() == 6.0
    assert selected[0, 2, 0, 0, 0].item() == 10.0
```

- [ ] **Step 4: Run and verify it passes**

Run: `pytest tests/models/test_vrt_fusion_integration.py::test_index_reducer_uses_configured_position -v`
Expected: PASS

- [ ] **Step 5: Write test for SpikeUpsample gradient flow**

Add to `tests/models/test_fusion_early_adapter.py`:

```python
def test_spike_upsample_gradient_flows():
    """SpikeUpsample refinement conv should have gradients after backward."""
    upsample = SpikeUpsample(spike_chans=4)
    spike = torch.randn(2, 4, 6, 4, requires_grad=True)
    out = upsample(spike, target_h=12, target_w=12)
    loss = out.sum()
    loss.backward()
    # Verify gradient flows through refinement conv
    for param in upsample.refine.parameters():
        assert param.grad is not None
        assert param.grad.abs().sum() > 0
```

- [ ] **Step 6: Run and verify it passes**

Run: `pytest tests/models/test_fusion_early_adapter.py::test_spike_upsample_gradient_flows -v`
Expected: PASS

- [ ] **Step 7: Run entire test suite**

Run: `pytest tests/models/test_fusion_early_adapter.py tests/models/test_vrt_fusion_integration.py tests/models/test_partial_loading.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add tests/models/test_fusion_early_adapter.py tests/models/test_vrt_fusion_integration.py
git commit -m "test: add integration tests for early fusion temporal expansion"
```

---

## Summary of Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| `SpikeUpsample` location | Inside `early.py` as submodule of `EarlyFusionAdapter` | Collocated with the only consumer; ~288 params (S=4) |
| `effective_in_chans` | Computed from `early_out_chans` when early fusion enabled | `conv_first` sees fusion output (3ch), not raw input (11ch) |
| Restoration reducer | Configurable `index`, `selector`, or `residual_selector` | Avoid hard-coding unverified `S//2`; allow zero-param baseline and learnable alternatives without changing pretrained VRT shapes |
| Partial loading | Key+shape matching (not positional zip) | Old `strict=False` was positionally pairing keys — broken for differently-structured models |
| `freeze_backbone` | Module-level function filtering by name substring | Simple, no special module registry needed; optimizer already filters by `requires_grad` |
| `output_mode` | Config-driven, default `'restoration'` | `interpolation` path available for future use; restoration is the current training target |
| Dual input spatial validation | Relaxed to only check `[B, T]` | In the current concat ingress, `_build_model_input_tensor()` resizes spike to RGB resolution before concat; `SpikeUpsample` remains as adapter-level alignment/refinement support for direct adapter tests and future non-concat routing |

## Notes on Existing Tests

The existing tests in `test_fusion_early_adapter.py` and `test_vrt_fusion_integration.py` use `out_chans=4` with `in_chans=4` (equal). These tests continue to work because:
- When `out_chans == in_chans`, the `effective_in_chans` computation produces the same result as before
- `spike_chans=0` (default) means no `SpikeUpsample` is created, so the spatial-same path is unchanged
- `S=1` default means no temporal reduction is needed, and the restoration reducer is bypassed

## Deferred Items

### Config Files Not Updated in This Plan

- `options/gopro_rgbspike_server.json` — same changes as `gopro_rgbspike_local.json`, deferred until server deployment needed
- `options/vrt_aligned.json` — new config for baseline comparison (dcn.type=dcnv2, fusion.enable=false), deferred until baseline experiment

### Stage C (LoRA) — Deferred

Stage C uses the same code paths with config switches:
- `train.freeze_backbone: false` — all params unfrozen
- `train.use_lora: true` — LoRA adapters injected into VRT attention QKV/proj

Implementation of LoRA injection is deferred to after Stage A validation succeeds. The config fields are defined now for forward compatibility.

### PASE Operator — Deferred

PASE's cross-bin temporal modeling is incompatible with temporal expansion (spike becomes 1-channel per frame). Requires separate "early-early fusion" design path. See spec for details.
