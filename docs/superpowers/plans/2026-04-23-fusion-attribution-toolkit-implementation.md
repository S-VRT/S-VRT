# Fusion Attribution Toolkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline, reusable fusion attribution toolkit for S-VRT that exports comparable heatmaps, fusion maps, panels, raw artifacts, and metadata for gated and future fusion operators.

**Architecture:** Add a small `scripts/analysis/fusion_attr/` package with pure helpers for samples, probes, targets, maps, perturbations, and panels, plus a thin CLI in `scripts/analysis/fusion_attribution.py`. Keep the tool outside the training loop; it reads an option file, checkpoint, and sample file, attaches inference-time probes, and writes artifacts under an output directory.

**Tech Stack:** Python, PyTorch, NumPy, OpenCV, argparse, JSON, pytest, optional pytorch-grad-cam import with an internal gradient fallback for tests.

---

### File Structure

Create these files:

- `scripts/analysis/fusion_attribution.py`  
  CLI entry point, argument parsing, config loading, model/data orchestration, artifact run loop.
- `scripts/analysis/fusion_attr/__init__.py`  
  Package exports for reusable analysis helpers.
- `scripts/analysis/fusion_attr/io.py`  
  JSON comment stripping, sample schema parsing, directory creation, metadata writing, tensor/image saving.
- `scripts/analysis/fusion_attr/probes.py`  
  Fusion adapter discovery, hook attachment, captured tensor records, optional operator `explain()` extraction.
- `scripts/analysis/fusion_attr/targets.py`  
  Mask creation, masked Charbonnier target, error maps, error reduction maps.
- `scripts/analysis/fusion_attr/maps.py`  
  Channel reductions, normalization, fusion delta, perturbation sensitivity, simple Grad-CAM fallback.
- `scripts/analysis/fusion_attr/perturb.py`  
  Spike/RGB perturbation helpers used by sensitivity maps.
- `scripts/analysis/fusion_attr/panels.py`  
  Six-column paper panel layout and image composition.
- `tests/analysis/test_fusion_attr_io.py`  
  Sample parsing and artifact path tests.
- `tests/analysis/test_fusion_attr_probes.py`  
  Generic probe and gated explanation tests.
- `tests/analysis/test_fusion_attr_targets_maps.py`  
  Masked target, error map, normalization, and sensitivity helper tests.
- `tests/analysis/test_fusion_attr_panels_cli.py`  
  Panel smoke and CLI help tests.

Modify these files:

- `models/fusion/operators/gated.py`  
  Add optional `explain()` support by storing detached-last forward internals without changing forward outputs.

Do not modify training loops, training losses, launch scripts, or dataset behavior.

---

### Task 1: Samples and Artifact IO

**Files:**
- Create: `scripts/analysis/fusion_attr/__init__.py`
- Create: `scripts/analysis/fusion_attr/io.py`
- Create: `tests/analysis/test_fusion_attr_io.py`

- [ ] **Step 1: Write failing tests for sample parsing and artifact paths**

Create `tests/analysis/test_fusion_attr_io.py`:

```python
import json
from pathlib import Path

import pytest

from scripts.analysis.fusion_attr.io import (
    AnalysisSample,
    build_sample_output_dir,
    load_samples_file,
    strip_json_comments,
    write_json,
)


def test_strip_json_comments_keeps_urls_and_removes_comments():
    text = '{"url": "http://example.test/a//b", /* block */ "value": 3 // line\n}'
    cleaned = strip_json_comments(text)
    data = json.loads(cleaned)
    assert data == {"url": "http://example.test/a//b", "value": 3}


def test_load_samples_file_parses_box_mask(tmp_path: Path):
    samples_path = tmp_path / "fusion_samples.json"
    samples_path.write_text(
        """
        {
          "samples": [
            {
              "clip": "GOPR0384_11_02",
              "frame": "001301",
              "frame_index": 3,
              "mask": {"type": "box", "xyxy": [1, 2, 9, 10], "label": "motion_boundary"},
              "reason": "fast motion edge"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    samples = load_samples_file(samples_path)
    assert samples == [
        AnalysisSample(
            clip="GOPR0384_11_02",
            frame="001301",
            frame_index=3,
            mask_type="box",
            xyxy=(1, 2, 9, 10),
            mask_label="motion_boundary",
            reason="fast motion edge",
        )
    ]


def test_load_samples_file_rejects_invalid_box(tmp_path: Path):
    samples_path = tmp_path / "fusion_samples.json"
    samples_path.write_text(
        '{"samples":[{"clip":"a","frame":"b","frame_index":0,"mask":{"type":"box","xyxy":[1,2,3]},"reason":"x"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="xyxy must contain four integers"):
        load_samples_file(samples_path)


def test_build_sample_output_dir_is_stable(tmp_path: Path):
    sample = AnalysisSample(
        clip="GOPR0384_11_02",
        frame="001301",
        frame_index=3,
        mask_type="box",
        xyxy=(1, 2, 9, 10),
        mask_label="motion_boundary",
        reason="fast motion edge",
    )
    assert build_sample_output_dir(tmp_path, sample) == tmp_path / "samples" / "GOPR0384_11_02_001301"


def test_write_json_creates_parent_directory(tmp_path: Path):
    target = tmp_path / "nested" / "metadata.json"
    write_json(target, {"b": 2, "a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
pytest tests/analysis/test_fusion_attr_io.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.analysis.fusion_attr'`.

- [ ] **Step 3: Add package marker files**

Create `scripts/analysis/fusion_attr/__init__.py`:

```python
"""Reusable offline attribution helpers for S-VRT fusion analysis."""
```

- [ ] **Step 4: Implement IO helpers**

Create `scripts/analysis/fusion_attr/io.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class AnalysisSample:
    clip: str
    frame: str
    frame_index: int
    mask_type: str
    xyxy: tuple[int, int, int, int]
    mask_label: str
    reason: str


def strip_json_comments(text: str) -> str:
    result: list[str] = []
    i = 0
    in_string = False
    string_char = ""
    while i < len(text):
        ch = text[i]
        if in_string:
            result.append(ch)
            if ch == "\\" and i + 1 < len(text):
                result.append(text[i + 1])
                i += 2
                continue
            if ch == string_char:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            string_char = ch
            result.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "/":
                i += 2
                while i < len(text) and text[i] not in "\n\r":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue
        result.append(ch)
        i += 1
    return "".join(result)


def _require_str(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _parse_xyxy(values: Iterable[Any]) -> tuple[int, int, int, int]:
    parsed = tuple(int(v) for v in values)
    if len(parsed) != 4:
        raise ValueError("xyxy must contain four integers")
    x1, y1, x2, y2 = parsed
    if x2 <= x1 or y2 <= y1:
        raise ValueError("xyxy must satisfy x2 > x1 and y2 > y1")
    return parsed


def load_samples_file(path: str | Path) -> list[AnalysisSample]:
    sample_path = Path(path)
    data = json.loads(strip_json_comments(sample_path.read_text(encoding="utf-8")))
    raw_samples = data.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("samples must be a list")
    samples: list[AnalysisSample] = []
    for item in raw_samples:
        if not isinstance(item, dict):
            raise ValueError("each sample must be an object")
        mask = item.get("mask")
        if not isinstance(mask, dict):
            raise ValueError("sample mask must be an object")
        mask_type = str(mask.get("type", "")).strip().lower()
        if mask_type != "box":
            raise ValueError(f"unsupported mask type: {mask_type}")
        samples.append(
            AnalysisSample(
                clip=_require_str(item, "clip"),
                frame=_require_str(item, "frame"),
                frame_index=int(item.get("frame_index", 0)),
                mask_type=mask_type,
                xyxy=_parse_xyxy(mask.get("xyxy", [])),
                mask_label=str(mask.get("label", "box")),
                reason=_require_str(item, "reason"),
            )
        )
    return samples


def build_sample_output_dir(out_root: str | Path, sample: AnalysisSample) -> Path:
    return Path(out_root) / "samples" / f"{sample.clip}_{sample.frame}"


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest tests/analysis/test_fusion_attr_io.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/analysis/fusion_attr/__init__.py scripts/analysis/fusion_attr/io.py tests/analysis/test_fusion_attr_io.py
git commit -m "feat(analysis): add fusion attribution sample io"
```

---

### Task 2: Fusion Probes and Gated Explanation

**Files:**
- Create: `scripts/analysis/fusion_attr/probes.py`
- Create: `tests/analysis/test_fusion_attr_probes.py`
- Modify: `models/fusion/operators/gated.py`

- [ ] **Step 1: Write failing probe tests**

Create `tests/analysis/test_fusion_attr_probes.py`:

```python
import torch
from torch import nn

from scripts.analysis.fusion_attr.probes import FusionProbe, find_fusion_adapter, reduce_operator_explanations
from models.fusion.operators.gated import GatedFusionOperator


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_adapter = nn.Identity()

    def forward(self, x):
        return self.fusion_adapter(x)


def test_find_fusion_adapter_prefers_named_attribute():
    net = TinyNet()
    assert find_fusion_adapter(net) is net.fusion_adapter


def test_fusion_probe_captures_inputs_and_output():
    module = nn.Identity()
    probe = FusionProbe(module)
    probe.attach()
    x = torch.randn(1, 2, 3, 4, 4)
    y = module(x)
    probe.close()
    record = probe.record
    assert record is not None
    assert torch.equal(record.output, y)
    assert torch.equal(record.inputs[0], x)
    assert record.module_name == "Identity"


def test_gated_operator_explain_exports_effective_update():
    op = GatedFusionOperator(rgb_chans=3, spike_chans=2, out_chans=3, operator_params={"hidden_chans": 4})
    rgb = torch.randn(1, 3, 6, 6)
    spike = torch.randn(1, 2, 6, 6)
    _ = op(rgb, spike)
    maps = op.explain()
    assert set(maps) == {"gate", "correction", "effective_update"}
    assert maps["gate"].shape == rgb.shape
    assert maps["effective_update"].shape == rgb.shape


def test_reduce_operator_explanations_converts_tensors_to_2d_maps():
    explanations = {
        "gate": torch.ones(1, 3, 4, 5),
        "effective_update": torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5),
    }
    reduced = reduce_operator_explanations(explanations)
    assert reduced["gate_mean"].shape == (4, 5)
    assert reduced["effective_update"].shape == (4, 5)
    assert reduced["gate_mean"].max().item() == 1.0
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
pytest tests/analysis/test_fusion_attr_probes.py -q
```

Expected: FAIL because `scripts.analysis.fusion_attr.probes` does not exist and `GatedFusionOperator.explain` is missing.

- [ ] **Step 3: Implement probe helpers**

Create `scripts/analysis/fusion_attr/probes.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class FusionProbeRecord:
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor
    module_name: str


def _detach_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (tuple, list)):
        return tuple(_detach_tensor(v) for v in value)
    return value


def find_fusion_adapter(model: nn.Module) -> nn.Module:
    if hasattr(model, "fusion_adapter"):
        return getattr(model, "fusion_adapter")
    if hasattr(model, "netG"):
        net = getattr(model, "netG")
        if hasattr(net, "fusion_adapter"):
            return getattr(net, "fusion_adapter")
    for _, module in model.named_modules():
        if module.__class__.__name__.lower().endswith("fusionadapter"):
            return module
    raise ValueError("Could not find fusion_adapter on model")


class FusionProbe:
    def __init__(self, module: nn.Module):
        self.module = module
        self.record: FusionProbeRecord | None = None
        self._handle = None

    def attach(self) -> None:
        self.close()
        self._handle = self.module.register_forward_hook(self._hook)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        tensor_output = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(tensor_output, torch.Tensor):
            return
        if tensor_output.requires_grad:
            tensor_output.retain_grad()
        tensor_inputs = tuple(v for v in _detach_tensor(inputs) if isinstance(v, torch.Tensor))
        self.record = FusionProbeRecord(
            inputs=tensor_inputs,
            output=tensor_output,
            module_name=module.__class__.__name__,
        )


def _channel_norm(tensor: torch.Tensor) -> torch.Tensor:
    data = tensor.detach().float()
    if data.ndim == 5:
        data = data[0, data.shape[1] // 2]
    elif data.ndim == 4:
        data = data[0]
    if data.ndim == 3:
        return torch.linalg.vector_norm(data, dim=0)
    if data.ndim == 2:
        return data
    raise ValueError(f"Expected 2D, 3D, 4D, or 5D tensor, got shape {tuple(tensor.shape)}")


def reduce_operator_explanations(explanations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    reduced: dict[str, torch.Tensor] = {}
    if "gate" in explanations:
        gate = explanations["gate"].detach().float()
        if gate.ndim == 4:
            reduced["gate_mean"] = gate[0].mean(dim=0)
        elif gate.ndim == 5:
            reduced["gate_mean"] = gate[0, gate.shape[1] // 2].mean(dim=0)
    if "correction" in explanations:
        reduced["correction_norm"] = _channel_norm(explanations["correction"])
    if "effective_update" in explanations:
        reduced["effective_update"] = _channel_norm(explanations["effective_update"])
    for name, value in explanations.items():
        if name not in {"gate", "correction", "effective_update"}:
            reduced[name] = _channel_norm(value)
    return reduced
```

- [ ] **Step 4: Add optional gated explanation internals**

Modify `models/fusion/operators/gated.py` by changing the two forward branches so they store the last gate/correction/effective update. In the 5D branch, replace:

```python
concat = torch.cat([rgb_flat, spike_flat], dim=1)
out = rgb_flat + self.gate(concat) * self.correction(concat)
return out.reshape(bsz, steps, self.out_chans, height, width)
```

with:

```python
concat = torch.cat([rgb_flat, spike_flat], dim=1)
gate = self.gate(concat)
correction = self.correction(concat)
effective_update = gate * correction
self._last_explain = {
    'gate': gate.reshape(bsz, steps, self.out_chans, height, width).detach(),
    'correction': correction.reshape(bsz, steps, self.out_chans, height, width).detach(),
    'effective_update': effective_update.reshape(bsz, steps, self.out_chans, height, width).detach(),
}
out = rgb_flat + effective_update
return out.reshape(bsz, steps, self.out_chans, height, width)
```

In the 4D branch, replace:

```python
concat = torch.cat([rgb_feat, spike_feat], dim=1)
return rgb_feat + self.gate(concat) * self.correction(concat)
```

with:

```python
concat = torch.cat([rgb_feat, spike_feat], dim=1)
gate = self.gate(concat)
correction = self.correction(concat)
effective_update = gate * correction
self._last_explain = {
    'gate': gate.detach(),
    'correction': correction.detach(),
    'effective_update': effective_update.detach(),
}
return rgb_feat + effective_update
```

Add this method before `__all__`:

```python
    def explain(self) -> Dict[str, torch.Tensor]:
        if not hasattr(self, '_last_explain'):
            return {}
        return dict(self._last_explain)
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest tests/analysis/test_fusion_attr_probes.py -q
pytest tests/models/test_fusion_factory.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/analysis/fusion_attr/probes.py tests/analysis/test_fusion_attr_probes.py models/fusion/operators/gated.py
git commit -m "feat(analysis): add fusion attribution probes"
```

---

### Task 3: Targets, Maps, and Perturbations

**Files:**
- Create: `scripts/analysis/fusion_attr/targets.py`
- Create: `scripts/analysis/fusion_attr/maps.py`
- Create: `scripts/analysis/fusion_attr/perturb.py`
- Create: `tests/analysis/test_fusion_attr_targets_maps.py`

- [ ] **Step 1: Write failing tests for masks, target, maps, and perturbations**

Create `tests/analysis/test_fusion_attr_targets_maps.py`:

```python
import torch

from scripts.analysis.fusion_attr.io import AnalysisSample
from scripts.analysis.fusion_attr.maps import (
    compute_error_map,
    compute_fusion_delta,
    normalize_map,
    reduce_to_2d,
)
from scripts.analysis.fusion_attr.perturb import perturb_spike
from scripts.analysis.fusion_attr.targets import build_box_mask, masked_charbonnier_target


def _sample():
    return AnalysisSample(
        clip="clip",
        frame="000001",
        frame_index=0,
        mask_type="box",
        xyxy=(1, 1, 3, 4),
        mask_label="box",
        reason="unit test",
    )


def test_build_box_mask_marks_expected_region():
    mask = build_box_mask(_sample(), height=5, width=6, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 5, 6)
    assert mask.sum().item() == 6
    assert mask[0, 0, 1:4, 1:3].sum().item() == 6


def test_masked_charbonnier_target_is_negative_loss():
    output = torch.ones(1, 3, 5, 6)
    gt = torch.zeros(1, 3, 5, 6)
    mask = build_box_mask(_sample(), 5, 6, output.device)
    target = masked_charbonnier_target(output, gt, mask, eps=1e-6)
    assert target.item() < 0


def test_reduce_to_2d_handles_5d_center_frame():
    tensor = torch.zeros(1, 5, 3, 4, 4)
    tensor[:, 2] = 2.0
    reduced = reduce_to_2d(tensor)
    assert reduced.shape == (4, 4)
    assert reduced.max().item() == 2.0 * (3 ** 0.5)


def test_normalize_map_percentile_clips_to_unit_range():
    values = torch.tensor([[0.0, 1.0], [2.0, 100.0]])
    out = normalize_map(values, low=0, high=100)
    assert out.min().item() == 0.0
    assert out.max().item() == 1.0


def test_compute_fusion_delta_uses_matching_shape():
    fused = torch.ones(1, 3, 4, 4)
    reference = torch.zeros(1, 3, 4, 4)
    delta = compute_fusion_delta(fused, reference)
    assert delta.shape == (4, 4)
    assert delta.max().item() == 3 ** 0.5


def test_compute_error_map_returns_mean_channel_abs_error():
    output = torch.ones(1, 3, 4, 4)
    gt = torch.zeros(1, 3, 4, 4)
    error = compute_error_map(output, gt)
    assert error.shape == (4, 4)
    assert error.max().item() == 1.0


def test_perturb_spike_zero_and_temporal_drop():
    spike = torch.ones(1, 4, 2, 3, 3)
    assert perturb_spike(spike, "zero").sum().item() == 0.0
    dropped = perturb_spike(spike, "temporal-drop")
    assert dropped[:, 2].sum().item() == 0.0
    assert dropped[:, 0].sum().item() > 0.0
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
pytest tests/analysis/test_fusion_attr_targets_maps.py -q
```

Expected: FAIL because target/map/perturb modules do not exist.

- [ ] **Step 3: Implement targets**

Create `scripts/analysis/fusion_attr/targets.py`:

```python
from __future__ import annotations

import torch

from .io import AnalysisSample


def build_box_mask(sample: AnalysisSample, height: int, width: int, device: torch.device) -> torch.Tensor:
    if sample.mask_type != "box":
        raise ValueError(f"Unsupported mask type: {sample.mask_type}")
    x1, y1, x2, y2 = sample.xyxy
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Mask box is empty after clipping")
    mask = torch.zeros(1, 1, height, width, device=device)
    mask[:, :, y1:y2, x1:x2] = 1.0
    return mask


def masked_charbonnier_target(
    output: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if output.shape != gt.shape:
        raise ValueError(f"output and gt shapes differ: {tuple(output.shape)} vs {tuple(gt.shape)}")
    while mask.ndim < output.ndim:
        mask = mask.unsqueeze(1)
    diff = (output - gt) * mask
    denom = mask.sum().clamp_min(1.0) * output.shape[-3]
    loss = torch.sqrt(diff * diff + eps).sum() / denom
    return -loss
```

- [ ] **Step 4: Implement maps**

Create `scripts/analysis/fusion_attr/maps.py`:

```python
from __future__ import annotations

import numpy as np
import torch


def reduce_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    data = tensor.detach().float()
    if data.ndim == 5:
        data = data[0, data.shape[1] // 2]
    elif data.ndim == 4:
        data = data[0]
    if data.ndim == 3:
        return torch.linalg.vector_norm(data, dim=0)
    if data.ndim == 2:
        return data
    raise ValueError(f"Expected 2D, 3D, 4D, or 5D tensor, got {tuple(tensor.shape)}")


def normalize_map(values: torch.Tensor, low: float = 1.0, high: float = 99.0) -> torch.Tensor:
    arr = values.detach().float().cpu().numpy()
    lo = float(np.percentile(arr, low))
    hi = float(np.percentile(arr, high))
    if hi <= lo:
        return torch.zeros_like(values, dtype=torch.float32)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return torch.from_numpy(out).to(dtype=torch.float32)


def compute_fusion_delta(fusion_output: torch.Tensor, rgb_reference: torch.Tensor) -> torch.Tensor:
    if fusion_output.shape != rgb_reference.shape:
        raise ValueError(
            f"fusion_output and rgb_reference shapes differ: {tuple(fusion_output.shape)} vs {tuple(rgb_reference.shape)}"
        )
    return reduce_to_2d(fusion_output - rgb_reference)


def compute_error_map(output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if output.shape != gt.shape:
        raise ValueError(f"output and gt shapes differ: {tuple(output.shape)} vs {tuple(gt.shape)}")
    data = (output.detach().float() - gt.detach().float()).abs()
    if data.ndim == 5:
        data = data[0, data.shape[1] // 2]
    elif data.ndim == 4:
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"Expected image tensor with channels, got {tuple(output.shape)}")
    return data.mean(dim=0)
```

- [ ] **Step 5: Implement perturbations**

Create `scripts/analysis/fusion_attr/perturb.py`:

```python
from __future__ import annotations

import torch


def perturb_spike(spike: torch.Tensor, mode: str) -> torch.Tensor:
    normalized = str(mode).strip().lower()
    if normalized == "zero":
        return torch.zeros_like(spike)
    if normalized == "noise":
        return torch.randn_like(spike) * spike.detach().float().std().clamp_min(1e-6)
    if normalized == "shuffle":
        flat = spike.reshape(-1)
        perm = torch.randperm(flat.numel(), device=spike.device)
        return flat[perm].reshape_as(spike)
    if normalized == "temporal-drop":
        out = spike.clone()
        if out.ndim < 5:
            raise ValueError("temporal-drop expects spike tensor [B,T,C,H,W]")
        out[:, out.shape[1] // 2] = 0
        return out
    raise ValueError(f"Unsupported spike perturbation mode: {mode}")
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
pytest tests/analysis/test_fusion_attr_targets_maps.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/analysis/fusion_attr/targets.py scripts/analysis/fusion_attr/maps.py scripts/analysis/fusion_attr/perturb.py tests/analysis/test_fusion_attr_targets_maps.py
git commit -m "feat(analysis): add fusion attribution maps"
```

---

### Task 4: Panel Rendering and Image Saving

**Files:**
- Modify: `scripts/analysis/fusion_attr/io.py`
- Create: `scripts/analysis/fusion_attr/panels.py`
- Create: `tests/analysis/test_fusion_attr_panels_cli.py`

- [ ] **Step 1: Write failing panel and image saving tests**

Create `tests/analysis/test_fusion_attr_panels_cli.py` with the panel tests first:

```python
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.analysis.fusion_attr.io import save_gray_map_png, save_rgb_tensor_png
from scripts.analysis.fusion_attr.panels import make_six_column_panel


def test_save_rgb_tensor_png_writes_bgr_file(tmp_path: Path):
    tensor = torch.zeros(3, 4, 5)
    tensor[0] = 1.0
    path = tmp_path / "rgb.png"
    save_rgb_tensor_png(path, tensor)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img.shape == (4, 5, 3)
    assert int(img[:, :, 2].max()) == 255


def test_save_gray_map_png_writes_uint8_file(tmp_path: Path):
    path = tmp_path / "map.png"
    save_gray_map_png(path, torch.ones(4, 5))
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (4, 5)


def test_make_six_column_panel_writes_panel(tmp_path: Path):
    images = {
        "Blurry RGB": np.zeros((12, 16, 3), dtype=np.uint8),
        "Spike cue": np.ones((12, 16, 3), dtype=np.uint8) * 20,
        "Restored": np.ones((12, 16, 3), dtype=np.uint8) * 40,
        "Error reduction": np.ones((12, 16, 3), dtype=np.uint8) * 60,
        "Attribution heatmap": np.ones((12, 16, 3), dtype=np.uint8) * 80,
        "Fusion-specific map": np.ones((12, 16, 3), dtype=np.uint8) * 100,
    }
    out = tmp_path / "panel.png"
    make_six_column_panel(out, images)
    panel = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert panel is not None
    assert panel.shape[0] > 12
    assert panel.shape[1] > 16 * 5
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
pytest tests/analysis/test_fusion_attr_panels_cli.py -q
```

Expected: FAIL because image saving and panel helpers do not exist.

- [ ] **Step 3: Add image saving helpers to IO**

Append to `scripts/analysis/fusion_attr/io.py`:

```python
import cv2
import numpy as np
import torch


def _ensure_chw_rgb(tensor: torch.Tensor) -> torch.Tensor:
    data = tensor.detach().float().cpu()
    if data.ndim == 5:
        data = data[0, data.shape[1] // 2]
    elif data.ndim == 4:
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got {tuple(tensor.shape)}")
    if data.shape[0] > 3:
        data = data[:3]
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)
    return data.clamp(0, 1)


def save_rgb_tensor_png(path: str | Path, tensor: torch.Tensor) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _ensure_chw_rgb(tensor)
    rgb = (data.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    bgr = rgb[:, :, ::-1]
    cv2.imwrite(str(target), bgr)


def save_gray_map_png(path: str | Path, tensor: torch.Tensor) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = tensor.detach().float().cpu().numpy()
    if data.ndim != 2:
        raise ValueError(f"Expected 2D map tensor, got {tuple(tensor.shape)}")
    if data.max() > data.min():
        data = (data - data.min()) / (data.max() - data.min())
    else:
        data = np.zeros_like(data)
    img = (data * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(target), img)
```

- [ ] **Step 4: Implement panel helper**

Create `scripts/analysis/fusion_attr/panels.py`:

```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


PANEL_COLUMNS = [
    "Blurry RGB",
    "Spike cue",
    "Restored",
    "Error reduction",
    "Attribution heatmap",
    "Fusion-specific map",
]


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    raise ValueError(f"Unsupported panel image shape: {image.shape}")


def make_six_column_panel(path: str | Path, images: dict[str, np.ndarray]) -> None:
    missing = [name for name in PANEL_COLUMNS if name not in images]
    if missing:
        raise ValueError(f"Missing panel columns: {missing}")
    cells = [_as_bgr(images[name]) for name in PANEL_COLUMNS]
    height = max(cell.shape[0] for cell in cells)
    width = max(cell.shape[1] for cell in cells)
    rendered = []
    for name, cell in zip(PANEL_COLUMNS, cells):
        resized = cv2.resize(cell, (width, height), interpolation=cv2.INTER_AREA)
        canvas = np.full((height + 28, width, 3), 255, dtype=np.uint8)
        canvas[:height] = resized
        cv2.putText(canvas, name, (4, height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        rendered.append(canvas)
    panel = np.concatenate(rendered, axis=1)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), panel)
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest tests/analysis/test_fusion_attr_panels_cli.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/analysis/fusion_attr/io.py scripts/analysis/fusion_attr/panels.py tests/analysis/test_fusion_attr_panels_cli.py
git commit -m "feat(analysis): add fusion attribution panels"
```

---

### Task 5: CLI Skeleton and Offline Run Manifest

**Files:**
- Create: `scripts/analysis/fusion_attribution.py`
- Modify: `tests/analysis/test_fusion_attr_panels_cli.py`

- [ ] **Step 1: Extend tests with CLI help and dry-run manifest**

Append to `tests/analysis/test_fusion_attr_panels_cli.py`:

```python
import json
import subprocess
import sys


def test_fusion_attribution_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/analysis/fusion_attribution.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--opt" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--samples" in result.stdout


def test_fusion_attribution_cli_dry_run_writes_manifest(tmp_path: Path):
    opt = tmp_path / "opt.json"
    samples = tmp_path / "samples.json"
    out = tmp_path / "out"
    opt.write_text('{"model":"vrt","netG":{"fusion":{"operator":"gated","placement":"early","mode":"replace"}}}', encoding="utf-8")
    samples.write_text(
        '{"samples":[{"clip":"clip","frame":"000001","frame_index":0,"mask":{"type":"box","xyxy":[0,0,2,2]},"reason":"unit"}]}',
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/fusion_attribution.py",
            "--opt",
            str(opt),
            "--checkpoint",
            "missing.pth",
            "--samples",
            str(samples),
            "--out",
            str(out),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry run complete" in result.stdout.lower()
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint"] == "missing.pth"
    assert manifest["num_samples"] == 1
```

- [ ] **Step 2: Run CLI tests and confirm dry-run fails**

Run:

```bash
pytest tests/analysis/test_fusion_attr_panels_cli.py -q
```

Expected: FAIL because `scripts/analysis/fusion_attribution.py` does not exist.

- [ ] **Step 3: Implement CLI skeleton with dry-run**

Create `scripts/analysis/fusion_attribution.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.fusion_attr.io import load_samples_file, strip_json_comments, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline S-VRT fusion attribution toolkit")
    parser.add_argument("--opt", required=True, help="Path to S-VRT option JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--samples", required=True, help="Path to fusion_samples.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--baseline-opt", default=None, help="Optional baseline option JSON")
    parser.add_argument("--baseline-checkpoint", default=None, help="Optional baseline checkpoint")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--cam-method", default="gradcam", choices=["gradcam", "hirescam", "fallback"])
    parser.add_argument("--target", default="masked_charbonnier", choices=["masked_charbonnier"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--center-frame-only", action="store_true")
    parser.add_argument("--save-raw", action="store_true", default=True)
    parser.add_argument("--save-panel", action="store_true", default=True)
    parser.add_argument("--perturb-spike", default="zero", choices=["zero", "shuffle", "noise", "temporal-drop"])
    parser.add_argument("--mask-source", default="manual", choices=["manual", "motion", "error-topk"])
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write manifest without loading model")
    return parser


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_run_manifest(args: argparse.Namespace, samples_count: int) -> None:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    opt_text = _read_text(args.opt)
    (out_root / "config_snapshot.json").write_text(strip_json_comments(opt_text), encoding="utf-8")
    write_json(
        out_root / "run_manifest.json",
        {
            "opt": args.opt,
            "checkpoint": args.checkpoint,
            "samples": args.samples,
            "num_samples": samples_count,
            "baseline_opt": args.baseline_opt,
            "baseline_checkpoint": args.baseline_checkpoint,
            "device": args.device,
            "cam_method": args.cam_method,
            "target": args.target,
            "perturb_spike": args.perturb_spike,
            "mask_source": args.mask_source,
            "dry_run": bool(args.dry_run),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    samples = load_samples_file(args.samples)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    write_run_manifest(args, len(samples))
    if args.dry_run:
        print("Fusion attribution dry run complete.")
        return 0
    raise RuntimeError("Model-backed attribution execution is added in the next task.")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests and commit**

Run:

```bash
pytest tests/analysis/test_fusion_attr_panels_cli.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/analysis/fusion_attribution.py tests/analysis/test_fusion_attr_panels_cli.py
git commit -m "feat(analysis): add fusion attribution cli skeleton"
```

---

### Task 6: Model-Backed Attribution Run and Final Verification

**Files:**
- Modify: `scripts/analysis/fusion_attribution.py`
- Modify: `scripts/analysis/fusion_attr/maps.py`
- Create: `docs/analysis/fusion_samples.example.json`
- Modify: `tests/analysis/test_fusion_attr_panels_cli.py`

- [ ] **Step 1: Add model-backed helper tests using a tiny fake runner**

Append to `tests/analysis/test_fusion_attr_panels_cli.py`:

```python
from scripts.analysis.fusion_attribution import select_center_frame_tensor


def test_select_center_frame_tensor_handles_5d_and_4d():
    video = torch.zeros(1, 5, 3, 4, 4)
    video[:, 2] = 7
    image = torch.ones(1, 3, 4, 4)
    assert select_center_frame_tensor(video).max().item() == 7
    assert select_center_frame_tensor(image).max().item() == 1
```

- [ ] **Step 2: Add a simple gradient CAM fallback**

Append to `scripts/analysis/fusion_attr/maps.py`:

```python
def gradient_activation_cam(activation: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if activation.grad is not None:
        activation.grad.zero_()
    target.backward(retain_graph=True)
    if activation.grad is None:
        raise RuntimeError("Activation gradient was not retained")
    grad = activation.grad.detach()
    act = activation.detach()
    if act.ndim == 5:
        weights = grad.mean(dim=(-1, -2), keepdim=True)
        cam = (weights * act).sum(dim=2)
        cam = cam[0, cam.shape[1] // 2]
    elif act.ndim == 4:
        weights = grad.mean(dim=(-1, -2), keepdim=True)
        cam = (weights * act).sum(dim=1)[0]
    else:
        raise ValueError(f"Expected 4D or 5D activation, got {tuple(act.shape)}")
    return torch.relu(cam.detach())
```

- [ ] **Step 3: Add center-frame helper and model-backed orchestration**

Modify `scripts/analysis/fusion_attribution.py`:

Add imports:

```python
import json
import shutil

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from scripts.analysis.fusion_attr.maps import (
    compute_error_map,
    compute_fusion_delta,
    gradient_activation_cam,
    normalize_map,
)
from scripts.analysis.fusion_attr.panels import make_six_column_panel
from scripts.analysis.fusion_attr.probes import FusionProbe, find_fusion_adapter, reduce_operator_explanations
from scripts.analysis.fusion_attr.targets import build_box_mask, masked_charbonnier_target
from scripts.analysis.fusion_attr.io import (
    build_sample_output_dir,
    save_gray_map_png,
    save_rgb_tensor_png,
)
```

Add helper functions before `main`:

```python
def select_center_frame_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 5:
        return tensor[:, tensor.shape[1] // 2]
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Expected 4D or 5D tensor, got {tuple(tensor.shape)}")


def _load_json_config(path: str) -> dict:
    return json.loads(strip_json_comments(Path(path).read_text(encoding="utf-8")))


def _prepare_eval_opt(opt: dict) -> dict:
    cfg = dict(opt)
    cfg["is_train"] = False
    cfg["dist"] = False
    cfg["rank"] = 0
    cfg.setdefault("path", {})
    return cfg


def _load_checkpoint_if_available(model, checkpoint: str) -> None:
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    bare = model.get_bare_model(model.netG) if hasattr(model, "get_bare_model") else model.netG
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "params" in state:
        state = state["params"]
    bare.load_state_dict(state, strict=False)


def _build_test_loader(opt: dict) -> DataLoader:
    datasets = opt.get("datasets", {})
    test_opt = dict(datasets.get("test") or datasets.get("val") or {})
    if not test_opt:
        raise ValueError("Option file must contain datasets.test or datasets.val for attribution")
    test_opt["phase"] = "test"
    dataset = define_Dataset(test_opt)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def _sample_matches(batch: dict, sample_clip: str) -> bool:
    folder = batch.get("folder")
    if isinstance(folder, (list, tuple)):
        folder = folder[0]
    if isinstance(folder, str) and sample_clip in folder:
        return True
    lq_path = batch.get("L_path") or batch.get("lq_path")
    if isinstance(lq_path, (list, tuple)) and lq_path:
        return sample_clip in str(lq_path[0])
    return False


def _find_batch_for_sample(loader: DataLoader, sample) -> dict:
    for batch in loader:
        if _sample_matches(batch, sample.clip):
            return batch
    raise ValueError(f"Could not find sample clip in dataset: {sample.clip}")
```

Replace the non-dry-run `RuntimeError` in `main` with:

```python
    opt = _prepare_eval_opt(_load_json_config(args.opt))
    model = define_Model(opt)
    _load_checkpoint_if_available(model, args.checkpoint)
    model.netG.eval()
    loader = _build_test_loader(opt)

    for sample in samples:
        batch = _find_batch_for_sample(loader, sample)
        lq = batch["L"].to(next(model.netG.parameters()).device)
        gt = batch["H"].to(lq.device)
        adapter = find_fusion_adapter(model.netG)
        probe = FusionProbe(adapter)
        probe.attach()
        model.netG.zero_grad(set_to_none=True)
        output = model.netG(lq)
        if isinstance(output, (tuple, list)):
            output = output[0]
        probe.close()
        if probe.record is None:
            raise RuntimeError("Fusion probe did not capture a forward pass")

        center_output = select_center_frame_tensor(output)
        center_gt = select_center_frame_tensor(gt)
        mask = build_box_mask(sample, center_output.shape[-2], center_output.shape[-1], center_output.device)
        activation = probe.record.output
        target = masked_charbonnier_target(center_output, center_gt, mask)
        cam = gradient_activation_cam(activation, target)

        sample_dir = build_sample_output_dir(args.out, sample)
        inputs_dir = sample_dir / "inputs"
        outputs_dir = sample_dir / "outputs"
        maps_dir = sample_dir / "maps"
        save_rgb_tensor_png(inputs_dir / "blurry_rgb.png", select_center_frame_tensor(lq[:, :, :3]))
        save_rgb_tensor_png(inputs_dir / "gt.png", center_gt)
        save_rgb_tensor_png(outputs_dir / "restored.png", center_output)
        error_full = compute_error_map(center_output, center_gt)
        save_gray_map_png(maps_dir / "error_full.png", normalize_map(error_full))
        save_gray_map_png(maps_dir / "cam.png", normalize_map(cam))
        np.save(str(maps_dir / "cam_raw.npy"), cam.detach().cpu().numpy())

        if probe.record.inputs:
            reference = probe.record.inputs[0]
            if reference.shape == probe.record.output.shape:
                fusion_delta = compute_fusion_delta(probe.record.output, reference)
                np.save(str(maps_dir / "fusion_delta.npy"), fusion_delta.cpu().numpy())
                save_gray_map_png(maps_dir / "fusion_delta.png", normalize_map(fusion_delta))

        operator = getattr(adapter, "operator", None)
        fusion_specific_path = maps_dir / "cam.png"
        if operator is not None and hasattr(operator, "explain"):
            reduced = reduce_operator_explanations(operator.explain())
            for name, value in reduced.items():
                np.save(str(maps_dir / f"{name}.npy"), value.detach().cpu().numpy())
                save_gray_map_png(maps_dir / f"{name}.png", normalize_map(value))
            if "effective_update" in reduced:
                fusion_specific_path = maps_dir / "effective_update.png"

        write_json(
            sample_dir / "metadata.json",
            {
                "sample_id": f"{sample.clip}_{sample.frame}",
                "frame_index": sample.frame_index,
                "mask_type": sample.mask_type,
                "mask_xyxy": list(sample.xyxy),
                "mask_label": sample.mask_label,
                "target": args.target,
                "cam_method": args.cam_method,
                "checkpoint": args.checkpoint,
                "opt": args.opt,
                "probe_module": probe.record.module_name,
            },
        )

        panel_images = {
            "Blurry RGB": cv2.imread(str(inputs_dir / "blurry_rgb.png"), cv2.IMREAD_COLOR),
            "Spike cue": cv2.imread(str(inputs_dir / "blurry_rgb.png"), cv2.IMREAD_COLOR),
            "Restored": cv2.imread(str(outputs_dir / "restored.png"), cv2.IMREAD_COLOR),
            "Error reduction": cv2.imread(str(maps_dir / "error_full.png"), cv2.IMREAD_COLOR),
            "Attribution heatmap": cv2.imread(str(maps_dir / "cam.png"), cv2.IMREAD_COLOR),
            "Fusion-specific map": cv2.imread(str(fusion_specific_path), cv2.IMREAD_COLOR),
        }
        make_six_column_panel(sample_dir / "panel.png", panel_images)
    print(f"Fusion attribution complete: {args.out}")
    return 0
```

- [ ] **Step 4: Add example samples file**

Create `docs/analysis/fusion_samples.example.json`:

```json
{
  "samples": [
    {
      "clip": "GOPR0384_11_02",
      "frame": "001301",
      "frame_index": 3,
      "mask": {
        "type": "box",
        "xyxy": [120, 80, 220, 160],
        "label": "motion_boundary"
      },
      "reason": "fast motion edge"
    }
  ]
}
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/analysis -q
```

Expected: PASS.

- [ ] **Step 6: Run existing fusion tests**

Run:

```bash
pytest tests/models/test_fusion_factory.py tests/models/test_fusion_early_adapter.py tests/models/test_fusion_middle_adapter.py -q
```

Expected: PASS.

- [ ] **Step 7: Run CLI help and dry run**

Run:

```bash
python scripts/analysis/fusion_attribution.py --help
```

Expected: command prints help including `--opt`, `--checkpoint`, `--samples`, and `--dry-run`.

Run:

```bash
python scripts/analysis/fusion_attribution.py --opt options/gopro_rgbspike_local_debug.json --checkpoint missing.pth --samples docs/analysis/fusion_samples.example.json --out .runtime/fusion_attr_dry --dry-run
```

Expected: command prints `Fusion attribution dry run complete.` and writes `.runtime/fusion_attr_dry/run_manifest.json`.

- [ ] **Step 8: Check formatting and commit**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

Commit:

```bash
git add scripts/analysis/fusion_attribution.py scripts/analysis/fusion_attr/maps.py docs/analysis/fusion_samples.example.json tests/analysis/test_fusion_attr_panels_cli.py
git commit -m "feat(analysis): run fusion attribution offline"
```

---

### Task 7: Documentation and Full Verification

**Files:**
- Modify: `docs/superpowers/specs/2026-04-23-fusion-attribution-toolkit-design.md`
- Modify: `docs/superpowers/plans/2026-04-23-fusion-attribution-toolkit-implementation.md`

- [ ] **Step 1: Run all new analysis tests**

Run:

```bash
pytest tests/analysis -q
```

Expected: PASS.

- [ ] **Step 2: Run relevant fusion regression tests**

Run:

```bash
pytest tests/models/test_fusion_factory.py tests/models/test_fusion_early_adapter.py tests/models/test_fusion_middle_adapter.py tests/models/test_vrt_fusion_integration.py -q
```

Expected: PASS. If `tests/models/test_vrt_fusion_integration.py` is absent in the current checkout, run `pytest tests/models -q -k fusion` and record the exact tests collected.

- [ ] **Step 3: Run CLI smoke commands**

Run:

```bash
python scripts/analysis/fusion_attribution.py --help
python scripts/analysis/fusion_attribution.py --opt options/gopro_rgbspike_local_debug.json --checkpoint missing.pth --samples docs/analysis/fusion_samples.example.json --out .runtime/fusion_attr_dry --dry-run
```

Expected: help prints successfully; dry run writes manifest and does not require a real checkpoint.

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` prints nothing. `git status --short` shows only intended files before the final commit.

- [ ] **Step 5: Commit verification documentation if changed**

If documentation files were updated during execution, commit them:

```bash
git add docs/superpowers/specs/2026-04-23-fusion-attribution-toolkit-design.md docs/superpowers/plans/2026-04-23-fusion-attribution-toolkit-implementation.md
git commit -m "docs: update fusion attribution implementation notes"
```

If no documentation files changed, skip this commit and report that no documentation commit was needed.

---

### Implementation Notes

- The first implementation may use the internal `gradient_activation_cam` fallback instead of depending on `pytorch-grad-cam`. This keeps tests deterministic and avoids adding an install-time dependency. A follow-up can wrap `pytorch-grad-cam` once the model-backed path is stable.
- Keep all analysis code under `scripts/analysis/`; do not import it from training code.
- Keep raw `.npy` maps and display-normalized `.png` maps separate.
- Do not describe CAM output as proof of performance in docs or captions. Use it as localization evidence paired with error maps and ablation metrics.
