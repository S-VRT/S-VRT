# Fusion Analysis Effective-Update / IG / PCA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing offline fusion attribution toolkit so the current trained checkpoints can export `mamba` effective-update maps, Integrated Gradients maps, and PCA-based feature analyses without any retraining.

**Architecture:** Reuse the existing `scripts/analysis/fusion_attribution.py` CLI and `scripts/analysis/fusion_attr/` helpers instead of building a second analysis path. Add an inference-time `explain()` cache to `MambaFusionOperator`, add pure-PyTorch Integrated Gradients and PCA helpers, and wire new CLI flags plus metadata/output files into the existing sample-based artifact layout.

**Tech Stack:** Python, PyTorch, NumPy, OpenCV, pytest, existing S-VRT fusion attribution toolkit

---

## File Map

- Modify: `models/fusion/operators/mamba.py`
  - Cache detached tensor-level explainability maps during forward.
  - Add `explain()` parity with `GatedFusionOperator`.
- Modify: `scripts/analysis/fusion_attr/probes.py`
  - Reduce the new `mamba` explanation tensors into 2D maps.
- Modify: `scripts/analysis/fusion_attr/maps.py`
  - Add pure-PyTorch Integrated Gradients helpers.
- Create: `scripts/analysis/fusion_attr/pca.py`
  - Add PCA projection, variance summary, and heatmap helpers using `torch.linalg.svd`.
- Modify: `scripts/analysis/fusion_attribution.py`
  - Add CLI switches for IG/PCA export and save new artifacts under each sample folder.
- Modify: `tests/analysis/test_fusion_attr_probes.py`
  - Cover `mamba.explain()` and reduction behavior.
- Modify: `tests/analysis/test_fusion_attr_targets_maps.py`
  - Cover Integrated Gradients map generation.
- Modify: `tests/analysis/test_fusion_attr_panels_cli.py`
  - Cover new CLI flags and PCA helper outputs.

### Task 1: Export tensor-level `mamba` explanations for offline visualization

**Files:**
- Modify: `models/fusion/operators/mamba.py`
- Modify: `tests/analysis/test_fusion_attr_probes.py`

- [ ] **Step 1: Write the failing `mamba.explain()` tests**

Append these tests to `tests/analysis/test_fusion_attr_probes.py`:

```python
from models.fusion.operators.mamba import MambaFusionOperator


def test_mamba_operator_explain_exports_effective_update_without_ssm_dependency():
    op = MambaFusionOperator(
        rgb_chans=3,
        spike_chans=1,
        out_chans=3,
        operator_params={
            "token_dim": 8,
            "token_stride": 2,
            "num_layers": 0,
            "enable_diagnostics": True,
        },
    )
    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)

    _ = op(rgb, spike)

    maps = op.explain()
    assert set(maps) >= {"gate", "delta", "effective_update", "token_energy"}
    assert maps["effective_update"].shape == rgb.shape
    assert maps["gate"].shape == rgb.shape


def test_reduce_operator_explanations_accepts_mamba_specific_maps():
    explanations = {
        "gate": torch.ones(1, 2, 3, 4, 5),
        "delta": torch.ones(1, 2, 3, 4, 5) * 2,
        "effective_update": torch.ones(1, 2, 3, 4, 5) * 3,
        "token_energy": torch.ones(1, 2, 1, 4, 5) * 4,
    }

    reduced = reduce_operator_explanations(explanations)

    assert reduced["gate_mean"].shape == (4, 5)
    assert reduced["delta"].shape == (4, 5)
    assert reduced["effective_update"].shape == (4, 5)
    assert reduced["token_energy"].shape == (4, 5)
```

- [ ] **Step 2: Run the focused probe tests and confirm they fail**

Run:

```bash
python -m pytest tests/analysis/test_fusion_attr_probes.py -q
```

Expected: FAIL because `MambaFusionOperator` does not yet implement `explain()`.

- [ ] **Step 3: Implement `mamba` explanation caching**

Update `models/fusion/operators/mamba.py`:

```python
self._last_explain: dict[str, torch.Tensor] | None = None
```

Inside `forward()`, after `effective_update` is computed, cache detached maps:

```python
token_energy = spike_tokens.detach().float().norm(dim=-1)
token_energy = token_energy.reshape(bsz, steps, token_h, token_w)
token_energy = F.interpolate(
    token_energy.reshape(bsz * steps, 1, token_h, token_w),
    size=(height, width),
    mode="bilinear",
    align_corners=False,
).reshape(bsz, steps, 1, height, width)

self._last_explain = {
    "gate": gate.detach(),
    "delta": delta.detach(),
    "effective_update": effective_update.detach(),
    "token_energy": token_energy.detach(),
}
```

Add the method:

```python
def explain(self) -> dict[str, torch.Tensor]:
    if self._last_explain is None:
        return {}
    return dict(self._last_explain)
```

No training behavior should change; this is inference-time cache only.

- [ ] **Step 4: Re-run the probe tests and confirm they pass**

Run:

```bash
python -m pytest tests/analysis/test_fusion_attr_probes.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion/operators/mamba.py tests/analysis/test_fusion_attr_probes.py
git commit -m "feat(analysis): export mamba explain maps"
```

### Task 2: Add pure-PyTorch Integrated Gradients for patch-targeted restoration attribution

**Files:**
- Modify: `scripts/analysis/fusion_attr/maps.py`
- Modify: `tests/analysis/test_fusion_attr_targets_maps.py`

- [ ] **Step 1: Write the failing Integrated Gradients tests**

Append these tests to `tests/analysis/test_fusion_attr_targets_maps.py`:

```python
from scripts.analysis.fusion_attr.maps import integrated_gradients_map


def test_integrated_gradients_map_returns_2d_heatmap_for_4d_input():
    class SumModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = SumModel()
    inp = torch.ones(1, 3, 4, 4, requires_grad=True)
    baseline = torch.zeros_like(inp)

    def target_fn(output):
        return output[:, :, :2, :2].sum()

    heatmap = integrated_gradients_map(model, inp, baseline, target_fn, steps=8)
    assert heatmap.shape == (4, 4)
    assert heatmap.max().item() > 0.0


def test_integrated_gradients_map_supports_tuple_inputs_and_input_index():
    class DualModel(torch.nn.Module):
        def forward(self, rgb, spike):
            return rgb + spike[:, :, : rgb.size(-2), : rgb.size(-1)]

    model = DualModel()
    rgb = torch.ones(1, 3, 4, 4, requires_grad=True)
    spike = torch.ones(1, 3, 4, 4, requires_grad=True)

    def target_fn(output):
        return output.sum()

    heatmap = integrated_gradients_map(
        model,
        (rgb, spike),
        (torch.zeros_like(rgb), torch.zeros_like(spike)),
        target_fn,
        steps=4,
        input_index=1,
    )
    assert heatmap.shape == (4, 4)
    assert heatmap.mean().item() > 0.0
```

- [ ] **Step 2: Run the map tests and confirm they fail**

Run:

```bash
python -m pytest tests/analysis/test_fusion_attr_targets_maps.py -q
```

Expected: FAIL because `integrated_gradients_map` does not exist.

- [ ] **Step 3: Implement Integrated Gradients helpers in `maps.py`**

Append these helpers to `scripts/analysis/fusion_attr/maps.py`:

```python
def _clone_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def integrated_gradients_map(
    model,
    inputs,
    baselines,
    target_fn,
    steps: int = 32,
    input_index: int = 0,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")

    tuple_input = isinstance(inputs, (tuple, list))
    if not tuple_input:
        inputs = (inputs,)
        baselines = (baselines,)

    if len(inputs) != len(baselines):
        raise ValueError("inputs and baselines must have the same arity")

    total_grad = None
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=inputs[input_index].device)[1:]

    for alpha in alphas:
        scaled_inputs = []
        tracked_tensor = None
        for idx, (inp, base) in enumerate(zip(inputs, baselines)):
            scaled = base + alpha * (inp - base)
            if idx == input_index:
                tracked_tensor = _clone_with_grad(scaled)
                scaled_inputs.append(tracked_tensor)
            else:
                scaled_inputs.append(scaled.detach())
        output = model(*scaled_inputs)
        target = target_fn(output)
        if target.ndim != 0:
            raise ValueError("target_fn must return a scalar tensor")
        grad = torch.autograd.grad(target, tracked_tensor, retain_graph=False, create_graph=False)[0]
        total_grad = grad if total_grad is None else total_grad + grad

    avg_grad = total_grad / float(steps)
    attr = (inputs[input_index] - baselines[input_index]).detach() * avg_grad.detach()
    return reduce_to_2d(attr.abs())
```

This keeps dependencies minimal and matches the current offline-analysis style.

- [ ] **Step 4: Re-run the map tests and confirm they pass**

Run:

```bash
python -m pytest tests/analysis/test_fusion_attr_targets_maps.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/fusion_attr/maps.py tests/analysis/test_fusion_attr_targets_maps.py
git commit -m "feat(analysis): add integrated gradients heatmaps"
```

### Task 3: Add PCA helpers for token/update feature summaries without `sklearn`

**Files:**
- Create: `scripts/analysis/fusion_attr/pca.py`
- Modify: `tests/analysis/test_fusion_attr_panels_cli.py`

- [ ] **Step 1: Write the failing PCA helper tests**

Append these tests to `tests/analysis/test_fusion_attr_panels_cli.py`:

```python
from scripts.analysis.fusion_attr.pca import (
    pca_feature_heatmap,
    pca_variance_ratio,
)


def test_pca_variance_ratio_sums_to_one_for_non_degenerate_input():
    feat = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    ratio = pca_variance_ratio(feat)
    assert ratio.ndim == 1
    assert ratio.sum().item() == pytest.approx(1.0, rel=1e-5)


def test_pca_feature_heatmap_returns_spatial_map():
    feat = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    heatmap = pca_feature_heatmap(feat)
    assert heatmap.shape == (3, 4)
```

- [ ] **Step 2: Run the panel/CLI tests and confirm they fail**

Run:

```bash
python -m pytest tests/analysis/test_fusion_attr_panels_cli.py -q
```

Expected: FAIL because `scripts.analysis.fusion_attr.pca` does not exist.

- [ ] **Step 3: Create the PCA helper module**

Create `scripts/analysis/fusion_attr/pca.py`:

```python
from __future__ import annotations

import torch


def _flatten_spatial_feature(feature: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    data = feature.detach().float()
    if data.ndim == 5:
        data = data[0, data.shape[1] // 2]
    elif data.ndim == 4:
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"Expected CHW feature tensor, got {tuple(feature.shape)}")
    channels, height, width = data.shape
    matrix = data.reshape(channels, height * width).transpose(0, 1)
    return matrix, height, width


def pca_variance_ratio(feature: torch.Tensor) -> torch.Tensor:
    matrix, _, _ = _flatten_spatial_feature(feature)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    total = energy.sum().clamp_min(1e-12)
    return energy / total


def pca_feature_heatmap(feature: torch.Tensor, component: int = 0) -> torch.Tensor:
    matrix, height, width = _flatten_spatial_feature(feature)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    if component < 0 or component >= vh.shape[0]:
        raise ValueError(f"component out of range: {component}")
    projection = centered @ vh[component].unsqueeze(1)
    return projection.reshape(height, width)
```

- [ ] **Step 4: Re-run the panel/CLI tests and confirm they pass**

Run:

```bash
python -m pytest tests/analysis/test_fusion_attr_panels_cli.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/fusion_attr/pca.py tests/analysis/test_fusion_attr_panels_cli.py
git commit -m "feat(analysis): add pca feature summaries"
```

### Task 4: Wire IG/PCA outputs into the offline attribution CLI and sample artifacts

**Files:**
- Modify: `scripts/analysis/fusion_attribution.py`
- Modify: `tests/analysis/test_fusion_attr_panels_cli.py`

- [ ] **Step 1: Write the failing CLI coverage tests**

Append this test to `tests/analysis/test_fusion_attr_panels_cli.py`:

```python
def test_fusion_attribution_cli_help_mentions_ig_and_pca():
    result = subprocess.run(
        [sys.executable, "scripts/analysis/fusion_attribution.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--ig-steps" in result.stdout
    assert "--save-ig" in result.stdout
    assert "--save-pca" in result.stdout
```

- [ ] **Step 2: Run the panel/CLI tests and confirm they fail**

Run:

```bash
python -m pytest tests/analysis/test_fusion_attr_panels_cli.py -q
```

Expected: FAIL because the new CLI flags are not present.

- [ ] **Step 3: Extend the CLI and artifact export**

Update `build_parser()` in `scripts/analysis/fusion_attribution.py`:

```python
parser.add_argument("--save-ig", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--ig-steps", type=int, default=32)
parser.add_argument("--save-pca", action=argparse.BooleanOptionalAction, default=False)
```

Import the new helpers:

```python
from scripts.analysis.fusion_attr.maps import (
    compute_error_map,
    compute_fusion_delta,
    gradient_activation_cam,
    integrated_gradients_map,
    normalize_map,
)
from scripts.analysis.fusion_attr.pca import pca_feature_heatmap, pca_variance_ratio
```

After the existing CAM export, add IG export for both RGB and spike views:

```python
if args.save_ig:
    rgb_input = select_center_frame_tensor(lq[:, :, :3]).detach()
    spike_input = select_center_frame_tensor(lq[:, :, 3:]).detach()

    def rgb_model(rgb_only):
        fused = torch.cat([rgb_only, spike_input], dim=1)
        out = model.netG(fused.unsqueeze(1))
        return select_center_frame_tensor(out if not isinstance(out, (tuple, list)) else out[0])

    def spike_model(spike_only):
        fused = torch.cat([rgb_input, spike_only], dim=1)
        out = model.netG(fused.unsqueeze(1))
        return select_center_frame_tensor(out if not isinstance(out, (tuple, list)) else out[0])

    def target_fn(restored):
        return masked_charbonnier_target(restored, center_gt, mask)

    ig_rgb = integrated_gradients_map(
        rgb_model,
        rgb_input,
        torch.zeros_like(rgb_input),
        target_fn,
        steps=args.ig_steps,
    )
    ig_spike = integrated_gradients_map(
        spike_model,
        spike_input,
        torch.zeros_like(spike_input),
        target_fn,
        steps=args.ig_steps,
    )
    save_gray_map_png(maps_dir / "ig_rgb.png", normalize_map(ig_rgb))
    save_gray_map_png(maps_dir / "ig_spike.png", normalize_map(ig_spike))
```

Then add PCA export when operator explanations are available:

```python
if args.save_pca and operator is not None and hasattr(operator, "explain"):
    reduced = reduce_operator_explanations(operator.explain())
    for name in ("effective_update", "delta", "token_energy"):
        if name in reduced:
            continue
    raw = operator.explain()
    for name in ("effective_update", "delta", "token_energy"):
        if name in raw:
            pca_map = pca_feature_heatmap(raw[name])
            np.save(str(maps_dir / f"{name}_pca.npy"), pca_map.detach().cpu().numpy())
            save_gray_map_png(maps_dir / f"{name}_pca.png", normalize_map(pca_map))
            variance = pca_variance_ratio(raw[name])[:3].detach().cpu().tolist()
            metadata.setdefault("pca_variance_ratio", {})[name] = variance
```

Keep the metadata write at the end, but build the metadata dict first so it can include optional `pca_variance_ratio`.

- [ ] **Step 4: Re-run the analysis test suite**

Run:

```bash
python -m pytest tests/analysis/test_fusion_attr_probes.py tests/analysis/test_fusion_attr_targets_maps.py tests/analysis/test_fusion_attr_panels_cli.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/fusion_attribution.py tests/analysis/test_fusion_attr_panels_cli.py
git add scripts/analysis/fusion_attr/maps.py scripts/analysis/fusion_attr/pca.py
git add models/fusion/operators/mamba.py tests/analysis/test_fusion_attr_probes.py tests/analysis/test_fusion_attr_targets_maps.py
git commit -m "feat(analysis): add ig and pca fusion exports"
```

## Self-Review

- Spec coverage: the plan covers all three approved deliverables: `mamba` effective-update export, Integrated Gradients maps, and PCA summaries.
- Placeholder scan: there are no `TODO`/`TBD` placeholders; every task includes exact files, test names, commands, and expected outcomes.
- Type consistency: the plan uses `explain()` for tensor export, `integrated_gradients_map()` for scalar-target attribution, and `pca_feature_heatmap()` / `pca_variance_ratio()` for PCA summaries consistently across tasks.
