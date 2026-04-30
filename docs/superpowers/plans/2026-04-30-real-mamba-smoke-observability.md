# Real Mamba Smoke Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CUDA-backed smoke coverage that verifies `dual_scale_temporal_mamba` with real `mamba_ssm`, including detailed shape and data-flow observations for operator, adapter, VRT, and config handoff.

**Architecture:** Keep the smoke test in `tests/smoke/` so it can be selected independently with `-m smoke`. The test uses small tensors and a tiny VRT instance, monkeypatching expensive backbone/flow internals only after the real fusion operator and adapter have run. Observations are asserted as structured dictionaries rather than relying on printed logs.

**Tech Stack:** Python, pytest, PyTorch CUDA, mamba_ssm, existing VRT/fusion adapter stack, existing option parser and dataset constructors.

---

## File Map

- Create: `tests/smoke/test_dual_scale_temporal_mamba_real_cuda.py`
  - Real CUDA smoke tests for operator internals, adapter data-flow metadata, tiny VRT forward behavior, and option/dataset construction.
- Modify: `pytest.ini`
  - Register `cuda` marker for strict pytest marker mode.
- Create: `docs/superpowers/plans/2026-04-30-real-mamba-smoke-observability.md`
  - This implementation handoff and checklist.

## Task 1: Add Real CUDA Smoke Tests

- [x] **Step 1: Write the CUDA smoke test file**

Create `tests/smoke/test_dual_scale_temporal_mamba_real_cuda.py` with tests that:

```python
import copy
import importlib.util

import pytest
import torch

from data.dataset_video_test import TrainDatasetRGBSpike as EvalDatasetRGBSpike
from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike
from models.architectures.vrt.vrt import VRT
from models.fusion.factory import create_fusion_adapter, create_fusion_operator
from utils import utils_option as option


pytestmark = [pytest.mark.smoke, pytest.mark.cuda]


def _require_real_mamba_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real dual_scale_temporal_mamba smoke coverage.")
    if importlib.util.find_spec("mamba_ssm") is None:
        pytest.skip("mamba_ssm is required for real dual_scale_temporal_mamba smoke coverage.")


def _tiny_operator_params():
    return {
        "token_dim": 8,
        "patch_stride": 4,
        "d_state": 8,
        "d_conv": 4,
        "expand": 1,
        "local_layers": 1,
        "global_layers": 1,
        "alpha_init": 0.05,
        "gate_bias_init": -2.0,
        "enable_diagnostics": True,
    }


def _make_operator(device):
    return create_fusion_operator(
        "dual_scale_temporal_mamba",
        rgb_chans=3,
        spike_chans=21,
        out_chans=3,
        operator_params=_tiny_operator_params(),
    ).to(device).eval()


def _make_inputs(device, *, batch=1, steps=3, height=16, width=16):
    torch.manual_seed(1234)
    rgb = torch.linspace(
        -0.5,
        0.5,
        steps=batch * steps * 3 * height * width,
        device=device,
        dtype=torch.float32,
    ).reshape(batch, steps, 3, height, width)
    spike = torch.rand(batch, steps, 21, height, width, device=device, dtype=torch.float32)
    return rgb, spike


def _shape(tensor):
    return tuple(tensor.shape)


def test_dual_scale_temporal_mamba_real_cuda_operator_observes_internal_flow():
    _require_real_mamba_cuda()
    device = torch.device("cuda:0")
    operator = _make_operator(device)
    rgb, spike = _make_inputs(device)

    observed = {}

    def capture(name):
        def hook(_module, inputs, output):
            observed[name] = {
                "input_shapes": [_shape(item) for item in inputs if torch.is_tensor(item)],
                "output_shape": _shape(output) if torch.is_tensor(output) else None,
                "output_mean": float(output.detach().float().mean().cpu()) if torch.is_tensor(output) else None,
            }

        return hook

    handles = [
        operator.spike_projector.register_forward_hook(capture("spike_projector")),
        operator.local_stage.register_forward_hook(capture("local_stage")),
        operator.summary_gate.register_forward_hook(capture("summary_gate")),
        operator.global_stage.register_forward_hook(capture("global_stage")),
        operator.rgb_context_encoder.register_forward_hook(capture("rgb_context_encoder")),
        operator.fusion_body.register_forward_hook(capture("fusion_body")),
        operator.delta_head.register_forward_hook(capture("delta_head")),
        operator.gate_head.register_forward_hook(capture("gate_head")),
    ]
    try:
        with torch.no_grad():
            out = operator(rgb, spike)
    finally:
        for handle in handles:
            handle.remove()

    assert out.shape == rgb.shape
    assert out.is_cuda
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, rgb)

    assert observed["spike_projector"]["input_shapes"] == [(63, 1, 16, 16)]
    assert observed["spike_projector"]["output_shape"] == (63, 8, 4, 4)
    assert observed["local_stage"]["input_shapes"] == [(48, 21, 8)]
    assert observed["local_stage"]["output_shape"] == (48, 21, 8)
    assert observed["summary_gate"]["input_shapes"] == [(48, 21, 8)]
    assert observed["summary_gate"]["output_shape"] == (48, 21, 1)
    assert observed["global_stage"]["input_shapes"] == [(16, 3, 8)]
    assert observed["global_stage"]["output_shape"] == (16, 3, 8)
    assert observed["rgb_context_encoder"]["input_shapes"] == [(3, 3, 16, 16)]
    assert observed["fusion_body"]["output_shape"] == (3, 8, 16, 16)
    assert observed["delta_head"]["output_shape"] == (3, 3, 16, 16)
    assert observed["gate_head"]["output_shape"] == (3, 3, 16, 16)

    diagnostics = operator.diagnostics()
    for key in ("local_norm", "global_norm", "summary_gate_mean", "effective_update_norm"):
        assert key in diagnostics
        assert isinstance(diagnostics[key], float)
        assert diagnostics[key] >= 0.0
    assert diagnostics["warmup_stage"] == "full"


def test_dual_scale_temporal_mamba_real_cuda_adapter_preserves_collapsed_data_flow():
    _require_real_mamba_cuda()
    device = torch.device("cuda:0")
    operator = _make_operator(device)
    adapter = create_fusion_adapter(
        placement="early",
        operator=operator,
        mode="replace",
        spike_chans=21,
        frame_contract="operator_default",
    ).to(device).eval()
    rgb, spike = _make_inputs(device, steps=2, height=16, width=16)

    with torch.no_grad():
        result = adapter(rgb=rgb, spike=spike)

    observations = {
        "rgb": _shape(rgb),
        "spike": _shape(spike),
        "fused_main": _shape(result["fused_main"]),
        "backbone_view": _shape(result["backbone_view"]),
        "aux_view": None if result["aux_view"] is None else _shape(result["aux_view"]),
        "meta": dict(result["meta"]),
    }

    assert observations["rgb"] == (1, 2, 3, 16, 16)
    assert observations["spike"] == (1, 2, 21, 16, 16)
    assert observations["fused_main"] == (1, 2, 3, 16, 16)
    assert observations["backbone_view"] == (1, 2, 3, 16, 16)
    assert observations["aux_view"] is None
    assert result["fused_main"].data_ptr() == result["backbone_view"].data_ptr()
    assert observations["meta"]["frame_contract"] == "collapsed"
    assert observations["meta"]["spike_bins"] == 21
    assert observations["meta"]["main_steps"] == 2
    assert observations["meta"]["exec_steps"] == 2
    assert observations["meta"]["aux_steps"] is None
    assert observations["meta"]["local_norm"] > 0.0
    assert observations["meta"]["global_norm"] > 0.0
    assert observations["meta"]["effective_update_norm"] >= 0.0


def _dual_scale_vrt_opt():
    return {
        "netG": {
            "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 24},
            "fusion": {
                "placement": "early",
                "operator": "dual_scale_temporal_mamba",
                "out_chans": 3,
                "operator_params": _tiny_operator_params(),
            },
            "output_mode": "restoration",
        },
        "datasets": {
            "train": {
                "spike": {
                    "representation": "raw_window",
                    "raw_window_length": 21,
                    "reconstruction": {"type": "spikecv_tfp", "num_bins": 4},
                }
            }
        },
    }


def test_dual_scale_temporal_mamba_real_cuda_vrt_forward_records_data_flow(monkeypatch):
    _require_real_mamba_cuda()
    device = torch.device("cuda:0")
    model = VRT(
        upscale=1,
        in_chans=24,
        out_chans=3,
        img_size=[3, 16, 16],
        window_size=[3, 8, 8],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=_dual_scale_vrt_opt(),
    ).to(device).eval()

    dummy_flows = [torch.zeros(1, 2, 2, 16, 16, device=device)] * 4
    monkeypatch.setattr(model, "get_flows", lambda _x, flow_spike=None: (dummy_flows, dummy_flows))
    monkeypatch.setattr(
        model,
        "get_aligned_image_2frames",
        lambda _x, _fb, _ff: [
            torch.zeros(1, 3, model.backbone_in_chans * 4, 16, 16, device=device),
            torch.zeros(1, 3, model.backbone_in_chans * 4, 16, 16, device=device),
        ],
    )
    monkeypatch.setattr(model, "forward_features", lambda _x, *_args, **_kwargs: torch.zeros_like(_x))

    rgb, spike = _make_inputs(device, steps=3, height=16, width=16)
    x = torch.cat([rgb, spike], dim=2)
    with torch.no_grad():
        y = model(x)

    observations = {
        "input": _shape(x),
        "output": _shape(y),
        "fusion_main": _shape(model._last_fusion_main),
        "fusion_exec": _shape(model._last_fusion_exec),
        "fusion_aux": model._last_fusion_aux,
        "meta": dict(model._last_fusion_meta),
        "spike_bins": model._last_spike_bins,
    }

    assert observations["input"] == (1, 3, 24, 16, 16)
    assert observations["output"] == (1, 3, 3, 16, 16)
    assert observations["fusion_main"] == (1, 3, 3, 16, 16)
    assert observations["fusion_exec"] == (1, 3, 3, 16, 16)
    assert observations["fusion_aux"] is None
    assert observations["spike_bins"] == 21
    assert observations["meta"]["spike_representation"] == "raw_window"
    assert observations["meta"]["spike_window_length"] == 21
    assert observations["meta"]["effective_spike_channels"] == 21
    assert observations["meta"]["frame_contract"] == "collapsed"
    assert observations["meta"]["local_norm"] > 0.0
    assert observations["meta"]["global_norm"] > 0.0
    assert torch.isfinite(y).all()


def test_dual_scale_temporal_mamba_option_constructs_real_dataset_configs():
    _require_real_mamba_cuda()
    opt = option.parse("options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json", is_train=True)
    train_cfg = copy.deepcopy(opt["datasets"]["train"])
    test_cfg = copy.deepcopy(opt["datasets"]["test"])

    assert train_cfg["spike_channels"] == 21
    assert test_cfg["spike_channels"] == 21
    assert train_cfg["spike_flow"]["subframes"] == 21
    assert test_cfg["spike_flow"]["subframes"] == 21
    assert opt["netG"]["input"]["raw_ingress_chans"] == 24

    train_dataset = TrainDatasetRGBSpike(train_cfg)
    eval_dataset = EvalDatasetRGBSpike(test_cfg)

    assert train_dataset.spike_representation == "raw_window"
    assert eval_dataset.spike_representation == "raw_window"
    assert train_dataset.raw_window_length == 21
    assert eval_dataset.raw_window_length == 21
    assert train_dataset.spike_flow_subframes == 21
    assert eval_dataset.spike_flow_subframes == 21
```

- [x] **Step 2: Run the smoke test and confirm it fails before marker registration**

Run:

```bash
uv run pytest tests/smoke/test_dual_scale_temporal_mamba_real_cuda.py -v
```

Expected: FAIL during collection with strict marker error for unknown `cuda`.

## Task 2: Register Marker and Verify Smoke

- [x] **Step 1: Register the cuda marker**

Modify `pytest.ini` markers:

```ini
    cuda: Tests requiring CUDA-capable GPU execution
```

- [x] **Step 2: Run the real CUDA smoke test**

Run:

```bash
uv run pytest tests/smoke/test_dual_scale_temporal_mamba_real_cuda.py -v
```

Expected: PASS when CUDA and `mamba_ssm` are available. SKIP is acceptable only when the environment lacks CUDA or `mamba_ssm`; on the current host this should PASS.

- [x] **Step 3: Run focused regressions**

Run:

```bash
uv run pytest tests/models/test_dual_scale_temporal_mamba.py tests/models/test_vrt_fusion_integration.py -k "dual_scale_temporal_mamba" -v
uv run pytest tests/smoke/test_dual_scale_temporal_mamba_real_cuda.py tests/e2e/test_gopro_rgbspike_server_e2e.py -k "dual_scale_temporal_mamba" -v
```

Expected: PASS.

- [x] **Step 4: Commit**

```bash
git add pytest.ini tests/smoke/test_dual_scale_temporal_mamba_real_cuda.py docs/superpowers/plans/2026-04-30-real-mamba-smoke-observability.md
git commit -m "test(fusion): add real cuda dual-scale mamba smoke"
```

## Self-Review

- Spec coverage: covers real Mamba/CUDA execution, operator internal shape observations, adapter collapsed data flow, VRT metadata, and config-to-dataset construction.
- Placeholder scan: no TODO/TBD/deferred implementation language remains.
- Type consistency: all test helpers use existing `create_fusion_operator`, `create_fusion_adapter`, `VRT`, `TrainDatasetRGBSpike`, and option parser APIs.
