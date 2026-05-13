# SCFlow Collapse Subframe Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make collapsed early-fusion VRT use fine-grained SCFlow subframe motion by composing subframe flows into RGB frame-level flows.

**Architecture:** Keep the dataset and collapsed backbone contract unchanged. Add a VRT-level `spike_flow.collapse_policy` parser and route collapsed `[B,N*S,25,H,W]` SCFlow inputs through a subframe composition helper instead of averaging Spike windows. Preserve the current `mean_windows` behavior as the default for compatibility.

**Tech Stack:** PyTorch, VRT architecture code, SCFlow wrapper, pytest.

---

## File Structure

- Modify `models/architectures/vrt/vrt.py`
  - Parse `spike_flow.collapse_policy` from `opt.datasets.train` or `opt.datasets.test`.
  - Preserve `_align_flow_spike_to_fused_time_axis()` for `mean_windows`.
  - Add collapsed subframe dispatch in `get_flow_2frames()`.
  - Add helpers to compute expanded subframe SCFlow and compose adjacent subframe flows into frame-level flows.
- Modify `tests/models/test_vrt_fusion_integration.py`
  - Keep the existing collapsed mean behavior under explicit/default `mean_windows`.
  - Add deterministic tests for `compose_subframes`.
- Modify `options/gopro_rgbspike_server.json`
  - Add explicit `"collapse_policy": "compose_subframes"` for train and test `spike_flow` blocks when this experiment should use fine-grained collapsed flow.

## Task 1: Lock Compatibility Behavior

**Files:**
- Modify: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Rename the existing mean behavior test**

Change:

```python
def test_vrt_structured_early_mamba_collapses_subframe_flow_spike(monkeypatch):
```

to:

```python
def test_vrt_structured_early_mamba_mean_windows_collapses_subframe_flow_spike(monkeypatch):
```

Keep the existing body unchanged. This test currently asserts that `[B,N*S,25,H,W]` becomes `[B,N,25,H,W]` by `mean(dim=2)`.

- [ ] **Step 2: Run the renamed test**

Run:

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py::test_vrt_structured_early_mamba_mean_windows_collapses_subframe_flow_spike -q
```

Expected: PASS.

- [ ] **Step 3: Commit the test rename**

```bash
git add tests/models/test_vrt_fusion_integration.py
git commit -m "test: clarify scflow collapsed mean policy"
```

## Task 2: Add Collapse Policy Parsing

**Files:**
- Modify: `models/architectures/vrt/vrt.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Write a failing test for policy parsing**

Add this test near the existing VRT fusion config tests:

```python
def test_vrt_parses_scflow_collapse_policy_from_dataset_config():
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
        optical_flow={"module": "scflow", "checkpoint": None, "params": {}},
        opt={
            "datasets": {
                "train": {
                    "spike_flow": {
                        "representation": "encoding25",
                        "subframes": 4,
                        "collapse_policy": "compose_subframes",
                    }
                }
            },
            "netG": {
                "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 7},
                "fusion": {
                    "placement": "early",
                    "operator": "mamba",
                    "out_chans": 3,
                    "operator_params": {"model_dim": 8, "num_layers": 1},
                    "early": {"frame_contract": "collapsed"},
                },
                "output_mode": "restoration",
            },
        },
    )

    assert model.spike_flow_collapse_policy == "compose_subframes"
```

- [ ] **Step 2: Run the policy parsing test and verify failure**

Run:

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py::test_vrt_parses_scflow_collapse_policy_from_dataset_config -q
```

Expected: FAIL with `AttributeError` for `spike_flow_collapse_policy`.

- [ ] **Step 3: Implement policy parsing in VRT**

In `models/architectures/vrt/vrt.py`, add this helper method inside `class VRT` before `forward()`:

```python
    @staticmethod
    def _resolve_spike_flow_collapse_policy(opt):
        datasets_cfg = (opt or {}).get("datasets", {}) if isinstance(opt, dict) else {}
        for split_name in ("train", "test"):
            split_cfg = datasets_cfg.get(split_name, {}) if isinstance(datasets_cfg, dict) else {}
            spike_flow_cfg = split_cfg.get("spike_flow", {}) if isinstance(split_cfg, dict) else {}
            if isinstance(spike_flow_cfg, dict) and "collapse_policy" in spike_flow_cfg:
                policy = str(spike_flow_cfg.get("collapse_policy", "mean_windows")).strip().lower()
                break
        else:
            policy = "mean_windows"

        aliases = {
            "mean": "mean_windows",
            "mean_window": "mean_windows",
            "mean_windows": "mean_windows",
            "compose": "compose_subframes",
            "compose_subframes": "compose_subframes",
        }
        if policy not in aliases:
            raise ValueError(
                "Unsupported spike_flow.collapse_policy="
                f"{policy!r}; expected one of {sorted(set(aliases.values()))}."
            )
        return aliases[policy]
```

Then in `__init__`, after `self._last_spike_bins = None`, add:

```python
        self.spike_flow_collapse_policy = self._resolve_spike_flow_collapse_policy(opt)
```

- [ ] **Step 4: Run policy parsing tests**

Run:

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py::test_vrt_parses_scflow_collapse_policy_from_dataset_config tests/models/test_vrt_fusion_integration.py::test_vrt_structured_early_mamba_mean_windows_collapses_subframe_flow_spike -q
```

Expected: PASS.

- [ ] **Step 5: Commit policy parsing**

```bash
git add models/architectures/vrt/vrt.py tests/models/test_vrt_fusion_integration.py
git commit -m "feat: parse scflow collapse policy"
```

## Task 3: Compose Subframe Flows For Collapsed SCFlow

**Files:**
- Modify: `models/architectures/vrt/vrt.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Write a deterministic helper test for flow composition**

Add this test to `tests/models/test_vrt_fusion_integration.py`:

```python
def test_vrt_compose_adjacent_flows_accumulates_constant_translation():
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
        optical_flow={"module": "scflow", "checkpoint": None, "params": {}},
        opt={"netG": {"input": {"strategy": "concat", "mode": "concat", "raw_ingress_chans": 7}}},
    )

    flows = torch.zeros(1, 8, 2, 6, 6)
    flows[:, :, 0, :, :] = 1.0
    composed = model._compose_adjacent_flows(flows, start=2, end=6)

    assert composed.shape == (1, 2, 6, 6)
    assert torch.allclose(composed[:, 0], torch.full((1, 6, 6), 4.0))
    assert torch.allclose(composed[:, 1], torch.zeros(1, 6, 6))
```

- [ ] **Step 2: Run the helper test and verify failure**

Run:

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py::test_vrt_compose_adjacent_flows_accumulates_constant_translation -q
```

Expected: FAIL with `AttributeError` for `_compose_adjacent_flows`.

- [ ] **Step 3: Implement adjacent flow composition helper**

In `models/architectures/vrt/vrt.py`, add this method inside `class VRT` before `get_flows()`:

```python
    @staticmethod
    def _compose_adjacent_flows(flows, start, end):
        if end <= start:
            raise ValueError(f"Cannot compose empty flow range start={start}, end={end}.")
        if flows.ndim != 5:
            raise ValueError(f"Expected flows [B,T,2,H,W], got {tuple(flows.shape)}.")

        composed = flows[:, start, :, :, :]
        for idx in range(start + 1, end):
            composed = composed + flow_warp(
                flows[:, idx, :, :, :],
                composed.permute(0, 2, 3, 1),
            )
        return composed
```

- [ ] **Step 4: Run helper test**

Run:

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py::test_vrt_compose_adjacent_flows_accumulates_constant_translation -q
```

Expected: PASS.

- [ ] **Step 5: Commit helper**

```bash
git add models/architectures/vrt/vrt.py tests/models/test_vrt_fusion_integration.py
git commit -m "feat: add flow composition helper"
```

## Task 4: Route Collapsed Compose Policy Through SCFlow

**Files:**
- Modify: `models/architectures/vrt/vrt.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Write a failing test for collapsed compose dispatch**

Add this test:

```python
def test_vrt_structured_early_mamba_compose_subframes_keeps_expanded_flow_spike(monkeypatch):
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
        optical_flow={"module": "scflow", "checkpoint": None, "params": {}},
        opt={
            "datasets": {
                "train": {
                    "spike_flow": {
                        "representation": "encoding25",
                        "subframes": 4,
                        "collapse_policy": "compose_subframes",
                    }
                }
            },
            "netG": {
                "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 7},
                "fusion": {
                    "placement": "early",
                    "operator": "mamba",
                    "out_chans": 3,
                    "operator_params": {"model_dim": 8, "num_layers": 1},
                    "early": {"frame_contract": "collapsed"},
                },
                "output_mode": "restoration",
            },
        },
    )

    monkeypatch.setattr(model.fusion_adapter.operator, "forward", lambda rgb, spike: rgb)

    captured = {}
    dummy_flows = [
        torch.zeros(1, 5, 2, 8, 8),
        torch.zeros(1, 5, 2, 4, 4),
        torch.zeros(1, 5, 2, 2, 2),
        torch.zeros(1, 5, 2, 1, 1),
    ]

    def _fake_get_flows(_x, flow_spike=None):
        captured["x_shape"] = tuple(_x.shape)
        captured["flow_spike_shape"] = tuple(flow_spike.shape)
        return dummy_flows, dummy_flows

    def _fake_aligned(_x, _fb, _ff):
        bsz, steps, _, height, width = _x.shape
        chans = model.backbone_in_chans * 4
        return [
            torch.zeros(bsz, steps, chans, height, width),
            torch.zeros(bsz, steps, chans, height, width),
        ]

    monkeypatch.setattr(model, "get_flows", _fake_get_flows)
    monkeypatch.setattr(model, "get_aligned_image_2frames", _fake_aligned)
    monkeypatch.setattr(model, "forward_features", lambda _x, *_args, **_kwargs: torch.zeros_like(_x))

    x = torch.randn(1, 6, 7, 8, 8)
    flow_spike = torch.randn(1, 24, 25, 8, 8)

    with torch.no_grad():
        _ = model(x, flow_spike=flow_spike)

    assert captured["x_shape"] == (1, 6, 3, 8, 8)
    assert captured["flow_spike_shape"] == (1, 24, 25, 8, 8)
```

- [ ] **Step 2: Run dispatch test and verify failure**

Run:

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py::test_vrt_structured_early_mamba_compose_subframes_keeps_expanded_flow_spike -q
```

Expected: FAIL because `forward()` still averages `flow_spike` before `get_flows()`.

- [ ] **Step 3: Update forward-time flow alignment dispatch**

In `forward()`, replace:

```python
                flow_spike = self._align_flow_spike_to_fused_time_axis(
                    flow_spike=flow_spike,
                    fused_steps=backbone_view.size(1),
                    spike_bins=spike_bins,
                )
```

with:

```python
                if not (
                    self.spike_flow_collapse_policy == "compose_subframes"
                    and flow_spike is not None
                    and flow_spike.ndim == 5
                    and flow_spike.size(1) == backbone_view.size(1) * spike_bins
                    and backbone_view.size(1) == fused_main.size(1)
                ):
                    flow_spike = self._align_flow_spike_to_fused_time_axis(
                        flow_spike=flow_spike,
                        fused_steps=backbone_view.size(1),
                        spike_bins=spike_bins,
                    )
```

- [ ] **Step 4: Run dispatch tests**

Run:

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py::test_vrt_structured_early_mamba_compose_subframes_keeps_expanded_flow_spike tests/models/test_vrt_fusion_integration.py::test_vrt_structured_early_mamba_mean_windows_collapses_subframe_flow_spike -q
```

Expected: PASS.

- [ ] **Step 5: Write a failing test for frame-level composed output shapes**

Add this test:

```python
def test_vrt_get_flow_2frames_compose_subframes_returns_frame_level_flows(monkeypatch):
    model = VRT(
        upscale=1,
        in_chans=7,
        out_chans=3,
        img_size=[3, 8, 8],
        window_size=[3, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "scflow", "checkpoint": None, "params": {}},
        opt={
            "datasets": {
                "train": {
                    "spike_flow": {
                        "representation": "encoding25",
                        "subframes": 4,
                        "collapse_policy": "compose_subframes",
                    }
                }
            },
            "netG": {"input": {"strategy": "concat", "mode": "concat", "raw_ingress_chans": 7}},
        },
    )
    model._last_spike_bins = 4

    calls = []

    class FakeSpikeFlow(torch.nn.Module):
        input_type = "spike"

        def forward(self, a, b):
            calls.append((tuple(a.shape), tuple(b.shape)))
            steps = a.shape[0]
            height, width = a.shape[-2:]
            full = torch.zeros(steps, 2, height, width)
            full[:, 0, :, :] = 1.0
            return [
                full,
                torch.zeros(steps, 2, height // 2, width // 2),
                torch.zeros(steps, 2, height // 4, width // 4),
                torch.zeros(steps, 2, height // 8, width // 8),
            ]

    model.spynet = FakeSpikeFlow()

    x = torch.zeros(1, 3, 3, 8, 8)
    flow_spike = torch.zeros(1, 12, 25, 8, 8)
    flows_backward, flows_forward = model.get_flow_2frames(x, flow_spike=flow_spike)

    assert calls[0][0] == (11, 25, 8, 8)
    assert calls[1][0] == (11, 25, 8, 8)
    assert flows_backward[0].shape == (1, 2, 2, 8, 8)
    assert flows_forward[0].shape == (1, 2, 2, 8, 8)
    assert torch.allclose(flows_backward[0][:, :, 0], torch.full((1, 2, 8, 8), 4.0))
    assert torch.allclose(flows_forward[0][:, :, 0], torch.full((1, 2, 8, 8), 4.0))
```

- [ ] **Step 6: Run shape/composition test and verify failure**

Run:

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py::test_vrt_get_flow_2frames_compose_subframes_returns_frame_level_flows -q
```

Expected: FAIL because `get_flow_2frames()` rejects `flow_spike.size(1) != x.size(1)`.

- [ ] **Step 7: Implement collapsed subframe composition in `get_flow_2frames()`**

In `models/architectures/vrt/vrt.py`, add this helper before `get_flow_2frames()`:

```python
    def _compose_subframe_flow_sequence(self, flow_spike, frame_steps, spike_bins, h, w):
        bsz, sub_steps, c_flow, _, _ = flow_spike.shape
        if spike_bins <= 1:
            raise ValueError("compose_subframes requires spike_bins > 1.")
        if sub_steps != frame_steps * spike_bins:
            raise ValueError(
                "compose_subframes requires flow_spike temporal length to equal "
                f"frames*spike_bins, got flow_spike={tuple(flow_spike.shape)}, "
                f"frames={frame_steps}, spike_bins={spike_bins}."
            )

        x_1 = flow_spike[:, :-1, ...].reshape(-1, c_flow, h, w)
        x_2 = flow_spike[:, 1:, ...].reshape(-1, c_flow, h, w)

        flows_backward_raw = self.spynet(x_1, x_2)
        flows_forward_raw = self.spynet(x_2, x_1)

        if not isinstance(flows_backward_raw, (list, tuple)):
            flows_backward_raw = [flows_backward_raw]
        if not isinstance(flows_forward_raw, (list, tuple)):
            flows_forward_raw = [flows_forward_raw]

        anchor = spike_bins // 2
        flows_backward = []
        flows_forward = []

        for scale_idx, (flow_b, flow_f) in enumerate(zip(flows_backward_raw, flows_forward_raw)):
            h_i = max(h // (2 ** scale_idx), 1)
            w_i = max(w // (2 ** scale_idx), 1)
            flow_b = flow_b.view(bsz, sub_steps - 1, 2, h_i, w_i)
            flow_f = flow_f.view(bsz, sub_steps - 1, 2, h_i, w_i)

            composed_b = []
            composed_f = []
            for frame_idx in range(frame_steps - 1):
                start = frame_idx * spike_bins + anchor
                end = (frame_idx + 1) * spike_bins + anchor
                composed_b.append(self._compose_adjacent_flows(flow_b, start, end))
                composed_f.append(self._compose_adjacent_flows(flow_f, start, end))

            flows_backward.append(torch.stack(composed_b, dim=1))
            flows_forward.append(torch.stack(composed_f, dim=1))

        return flows_backward, flows_forward
```

Then in `get_flow_2frames()`, inside the SCFlow branch after channel/spatial validation and before the existing batch/time equality check, add:

```python
            spike_bins = int(getattr(self, "_last_spike_bins", 1) or 1)
            if (
                self.spike_flow_collapse_policy == "compose_subframes"
                and spike_bins > 1
                and flow_spike.size(1) == n * spike_bins
            ):
                return self._compose_subframe_flow_sequence(
                    flow_spike=flow_spike,
                    frame_steps=n,
                    spike_bins=spike_bins,
                    h=h,
                    w=w,
                )
```

Keep the existing `flow_spike.size(0) != b or flow_spike.size(1) != n` error path after this new early return.

- [ ] **Step 8: Run composition tests**

Run:

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py::test_vrt_get_flow_2frames_compose_subframes_returns_frame_level_flows tests/models/test_vrt_fusion_integration.py::test_vrt_compose_adjacent_flows_accumulates_constant_translation -q
```

Expected: PASS.

- [ ] **Step 9: Run SCFlow/fusion regression tests**

Run:

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py::test_vrt_structured_early_mamba_compose_subframes_keeps_expanded_flow_spike tests/models/test_vrt_fusion_integration.py::test_vrt_structured_early_mamba_mean_windows_collapses_subframe_flow_spike tests/models/test_optical_flow_scflow_contract.py tests/models/test_optical_flow_scflow_integration.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit composition routing**

```bash
git add models/architectures/vrt/vrt.py tests/models/test_vrt_fusion_integration.py
git commit -m "feat: compose scflow subframes in collapsed fusion"
```

## Task 5: Enable Policy In Server Config

**Files:**
- Modify: `options/gopro_rgbspike_server.json`

- [ ] **Step 1: Add explicit collapse policy to train and test `spike_flow` blocks**

Change each existing block from:

```json
"spike_flow": {
  "representation": "encoding25",
  "dt": 10,
  "root": "auto",
  "subframes": 4
}
```

to:

```json
"spike_flow": {
  "representation": "encoding25",
  "dt": 10,
  "root": "auto",
  "subframes": 4,
  "collapse_policy": "compose_subframes"
}
```

- [ ] **Step 2: Validate JSON config parsing**

Run:

```bash
uv run python -c 'from utils import utils_option as option; opt = option.parse("options/gopro_rgbspike_server.json", is_train=True); print(opt["datasets"]["train"]["spike_flow"]["collapse_policy"]); print(opt["datasets"]["test"]["spike_flow"]["collapse_policy"])'
```

Expected output:

```text
compose_subframes
compose_subframes
```

- [ ] **Step 3: Commit config**

```bash
git add options/gopro_rgbspike_server.json
git commit -m "config: enable composed scflow collapse policy"
```

## Task 6: Final Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py tests/models/test_optical_flow_scflow_contract.py tests/models/test_optical_flow_scflow_integration.py -q
```

Expected: PASS.

- [ ] **Step 2: Check worktree**

Run:

```bash
git status --short
```

Expected: no modified tracked files. Untracked local artifacts such as `weights/vrt/` may remain and should not be committed unless explicitly requested.
