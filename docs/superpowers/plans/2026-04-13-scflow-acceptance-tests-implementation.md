# SCFlow Acceptance Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ~30 acceptance tests across two layers to verify the SCFlow strict semantic integration.

**Architecture:** Layer 1 extends the existing contract test file with unit-level boundary/error-path tests. Layer 2 creates a new integration test file with synthetic-data functional tests covering the full encoding25 → dataset → model → VRT → wrapper data flow.

**Tech Stack:** Python, pytest, PyTorch, numpy, existing S-VRT test helpers. Environment managed with uv; run tests via `.venv/bin/python -m pytest`.

---

## Task 1: Layer 1 — Contract test additions

**Modify:** `tests/models/test_optical_flow_scflow_contract.py`

**Run command:**
```bash
.venv/bin/python -m pytest tests/models/test_optical_flow_scflow_contract.py -v
```
**Expected:** all tests pass (existing 8 + new ~26 = ~34 total)

**Commit:** `test(contract): add comprehensive encoding25/wrapper/model/vrt/dataset contract tests`

### Steps

- [ ] Add imports at the top of the file (after existing imports):

```python
from data.spike_recc.encoding25 import (
    build_output_dir,
    compute_center_index,
    validate_center_bounds,
    build_centered_window,
    validate_encoding25_tensor,
)
```

- [ ] Add Group A — encoding25 utility contracts (all `@pytest.mark.unit`):

```python
@pytest.mark.unit
def test_validate_encoding25_tensor_rejects_2d_input():
    with pytest.raises(ValueError, match="ndim"):
        validate_encoding25_tensor(np.zeros((16, 16)))


@pytest.mark.unit
def test_validate_encoding25_tensor_rejects_4d_input():
    with pytest.raises(ValueError, match="ndim"):
        validate_encoding25_tensor(np.zeros((25, 1, 16, 16)))


@pytest.mark.unit
def test_validate_encoding25_tensor_accepts_valid():
    validate_encoding25_tensor(np.zeros((25, 16, 16), dtype=np.float32))  # no error


@pytest.mark.unit
def test_build_output_dir_rejects_zero_dt(tmp_path):
    with pytest.raises(ValueError, match="dt"):
        build_output_dir(tmp_path / "clip", dt=0)


@pytest.mark.unit
def test_build_output_dir_rejects_negative_dt(tmp_path):
    with pytest.raises(ValueError, match="dt"):
        build_output_dir(tmp_path / "clip", dt=-1)


@pytest.mark.unit
def test_compute_center_index_formula():
    # center_offset + (frame_index - clip_start_frame) * dt = 40 + 3*10 = 70
    result = compute_center_index(frame_index=5, clip_start_frame=2, dt=10, center_offset=40)
    assert result == 70


@pytest.mark.unit
def test_validate_center_bounds_rejects_left_boundary():
    # center=50, edge_margin=40: center - 12 = 38 < 40 → ValueError
    with pytest.raises(ValueError):
        validate_center_bounds(center=50, total_length=500, edge_margin=40)


@pytest.mark.unit
def test_validate_center_bounds_rejects_right_boundary():
    # center=448, total_length=500, edge_margin=40: center+12=460 >= 500-40=460 → ValueError
    with pytest.raises(ValueError):
        validate_center_bounds(center=448, total_length=500, edge_margin=40)


@pytest.mark.unit
def test_validate_center_bounds_accepts_valid_center():
    # center=250, total_length=500, edge_margin=40: 250-12=238>=40, 250+12=262<460 → ok
    validate_center_bounds(center=250, total_length=500, edge_margin=40)  # no error


@pytest.mark.unit
def test_build_centered_window_rejects_2d_spike_matrix():
    with pytest.raises(ValueError, match="spike_matrix"):
        build_centered_window(np.zeros((100, 16)), center=50)


@pytest.mark.unit
def test_build_centered_window_rejects_length_not_25():
    with pytest.raises(ValueError, match="length=25"):
        build_centered_window(np.zeros((100, 8, 8), dtype=np.float32), center=50, length=24)


@pytest.mark.unit
def test_build_centered_window_rejects_center_too_close_to_start():
    # center=5: st=5-12=-7 < 0 → ValueError
    with pytest.raises(ValueError, match="center"):
        build_centered_window(np.zeros((100, 8, 8), dtype=np.float32), center=5)


@pytest.mark.unit
def test_build_centered_window_rejects_center_too_close_to_end():
    # center=95, T=100: ed=95+12+1=108 > 100 → ValueError
    with pytest.raises(ValueError, match="center"):
        build_centered_window(np.zeros((100, 8, 8), dtype=np.float32), center=95)
```

- [ ] Add Group B — SCFlowWrapper contracts (`@pytest.mark.unit`):

```python
@pytest.mark.unit
def test_scflow_wrapper_rejects_3d_input():
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    with pytest.raises(ValueError, match="ndim=4"):
        wrapper._validate_spike_pair(torch.randn(25, 8, 8), torch.randn(1, 25, 8, 8))


@pytest.mark.unit
def test_scflow_wrapper_rejects_5d_input():
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    with pytest.raises(ValueError, match="ndim=4"):
        wrapper._validate_spike_pair(torch.randn(1, 1, 25, 8, 8), torch.randn(1, 25, 8, 8))


@pytest.mark.unit
def test_scflow_wrapper_rejects_wrong_channels_on_spk2():
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    with pytest.raises(ValueError, match="channels=25"):
        wrapper._validate_spike_pair(torch.randn(1, 25, 8, 8), torch.randn(1, 11, 8, 8))
```

- [ ] Add Group C — ModelPlain contracts (`@pytest.mark.unit`):

```python
@pytest.mark.unit
def test_model_plain_rejects_l_flow_spike_ndim_4():
    model = _build_stub_plain_model(flow_module="scflow")
    data = {
        "L": torch.randn(1, 4, 7, 8, 8),
        "L_flow_spike": torch.randn(1, 25, 8, 8),  # ndim=4, should be 5
    }
    with pytest.raises(ValueError, match="L_flow_spike"):
        model.feed_data(data, need_H=False)


@pytest.mark.unit
def test_model_plain_rejects_l_flow_spike_wrong_channels():
    model = _build_stub_plain_model(flow_module="scflow")
    data = {
        "L": torch.randn(1, 4, 7, 8, 8),
        "L_flow_spike": torch.randn(1, 4, 11, 8, 8),  # channels=11, not 25
    }
    with pytest.raises(ValueError, match="L_flow_spike"):
        model.feed_data(data, need_H=False)


@pytest.mark.unit
def test_model_plain_stores_l_flow_spike_on_valid_input():
    model = _build_stub_plain_model(flow_module="scflow")
    data = {
        "L": torch.randn(1, 4, 7, 8, 8),
        "L_flow_spike": torch.randn(1, 4, 25, 8, 8),
    }
    model.feed_data(data, need_H=False)
    assert model.L_flow_spike is not None
    assert tuple(model.L_flow_spike.shape) == (1, 4, 25, 8, 8)


@pytest.mark.unit
def test_model_plain_flow_module_alias_spike_flow():
    model = _build_stub_plain_model(flow_module="spike_flow")
    assert model._flow_module_name() == "scflow"


@pytest.mark.unit
def test_model_plain_clears_l_flow_spike_for_non_scflow():
    model = _build_stub_plain_model(flow_module="spynet")
    data = {
        "L": torch.randn(1, 4, 7, 8, 8),
    }
    model.feed_data(data, need_H=False)
    assert model.L_flow_spike is None
```

- [ ] Add Group D — VRT contracts (`@pytest.mark.unit`):

```python
@pytest.mark.unit
def test_vrt_rejects_flow_spike_ndim_4():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    # pass ndim=4 tensor instead of required ndim=5
    with pytest.raises(ValueError, match="ndim"):
        vrt.get_flow_2frames(x, flow_spike=torch.randn(4, 25, 16, 16))


@pytest.mark.unit
def test_vrt_rejects_flow_spike_batch_mismatch():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    with pytest.raises(ValueError, match="B,T"):
        vrt.get_flow_2frames(x, flow_spike=torch.randn(2, 4, 25, 16, 16))


@pytest.mark.unit
def test_vrt_rejects_flow_spike_time_mismatch():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    with pytest.raises(ValueError, match="B,T"):
        vrt.get_flow_2frames(x, flow_spike=torch.randn(1, 3, 25, 16, 16))


@pytest.mark.unit
def test_vrt_rejects_flow_spike_spatial_mismatch():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    with pytest.raises(ValueError, match="spatial"):
        vrt.get_flow_2frames(x, flow_spike=torch.randn(1, 4, 25, 8, 8))


@pytest.mark.unit
def test_vrt_rejects_flow_spike_wrong_channels():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    with pytest.raises(ValueError, match="channels=25"):
        vrt.get_flow_2frames(x, flow_spike=torch.randn(1, 4, 11, 16, 16))
```

- [ ] Add Group E — Dataset contracts (`@pytest.mark.unit`):

```python
@pytest.mark.unit
def test_dataset_no_spike_flow_config_disables_encoding25():
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds._parse_spike_flow_config({}, optical_flow_module="")
    assert ds.use_encoding25_flow is False


@pytest.mark.unit
def test_dataset_load_path_construction_auto_mode(tmp_path):
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds.spike_root = tmp_path
    ds.spike_flow_root = "auto"
    ds.spike_flow_dt = 10
    ds.filename_tmpl = "06d"
    # artifact does not exist; we just check the error message contains tmp_path
    with pytest.raises(ValueError, match=str(tmp_path)):
        ds._load_encoded_flow_spike("clip_a", 1)


@pytest.mark.unit
def test_dataset_load_path_construction_explicit_root(tmp_path):
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds.spike_root = tmp_path / "other"
    ds.spike_flow_root = "/explicit/root"
    ds.spike_flow_dt = 10
    ds.filename_tmpl = "06d"
    with pytest.raises(ValueError, match="/explicit/root"):
        ds._load_encoded_flow_spike("clip_a", 1)
```

---

## Task 2: Layer 2 — Integration test file

**Create:** `tests/models/test_optical_flow_scflow_integration.py`

**Run command:**
```bash
.venv/bin/python -m pytest tests/models/test_optical_flow_scflow_integration.py -v
```
**Expected:** all 11 integration tests pass

**Commit:** `test(integration): add scflow functional integration tests`

### Steps

- [ ] Create `tests/models/test_optical_flow_scflow_integration.py` with the file header, imports, and shared stubs:

```python
"""Layer 2: SCFlow functional integration tests using synthetic data."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike
from data.spike_recc.encoding25 import (
    build_centered_window,
    compute_center_index,
    validate_encoding25_tensor,
)
from models.architectures.vrt.vrt import VRT
from models.model_plain import ModelPlain
from models.optical_flow.scflow.wrapper import SCFlowWrapper


# ---------------------------------------------------------------------------
# Shared stubs (defined here; do NOT import from contract test file)
# ---------------------------------------------------------------------------

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _DummyTimer:
    def timer(self, _name): return _NullCtx()


class _DummySpikeFlow:
    input_type = "spike"

    def __init__(self):
        self.calls = []

    def __call__(self, x1, x2):
        self.calls.append((x1, x2))
        bsz, _c, h, w = x1.shape
        return [torch.zeros(bsz, 2, h // (2**i), w // (2**i)) for i in range(4)]


class _MockSCFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.received_dt = None

    def forward(self, spk1, spk2, flow_init, dt=10):
        self.received_dt = dt
        b, _, h, w = spk1.shape
        flows = [torch.zeros(b, 2, h // (2**i), w // (2**i)) for i in range(4)]
        return flows, {}


class _RecordingNet:
    training = True
    _input_path_marker = None

    def set_input_path_marker(self, m):
        self._input_path_marker = m

    def __call__(self, x, **kwargs):
        self.last_call_kwargs = kwargs
        return torch.zeros_like(x[:, :, :3, :, :])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_stub_plain_model(flow_module="scflow"):
    model = ModelPlain.__new__(ModelPlain)
    model.opt = {
        "netG": {
            "in_chans": 7,
            "input_mode": "concat",
            "optical_flow": {"module": flow_module},
        }
    }
    model.device = "cpu"
    model.timer = _DummyTimer()
    model.netG = _RecordingNet()
    model.L_flow_spike = None
    return model


def _build_stub_plain_model_with_net(flow_module, net):
    model = ModelPlain.__new__(ModelPlain)
    model.opt = {
        "netG": {
            "in_chans": 7,
            "input_mode": "concat",
            "optical_flow": {"module": flow_module},
        }
    }
    model.device = "cpu"
    model.timer = _DummyTimer()
    model.netG = net
    model.L_flow_spike = None
    return model
```

- [ ] Add Group 1 — encoding25 data round-trip tests (`@pytest.mark.integration`):

```python
# ---------------------------------------------------------------------------
# Group 1 — encoding25 data round-trip
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_build_centered_window_extracts_correct_slice():
    spike_matrix = np.arange(100 * 8 * 8).reshape(100, 8, 8).astype(np.float32)
    center = 50
    result = build_centered_window(spike_matrix, center=center)
    expected = spike_matrix[38:63]
    assert np.array_equal(result, expected)


@pytest.mark.integration
def test_compute_center_and_build_window_pipeline():
    spike_matrix = np.zeros((200, 8, 8), dtype=np.float32)
    center = compute_center_index(frame_index=3, clip_start_frame=0, dt=10, center_offset=40)
    assert center == 70
    result = build_centered_window(spike_matrix, center=center)
    expected = spike_matrix[58:83]
    assert np.array_equal(result, expected)


@pytest.mark.integration
def test_encoding25_npy_roundtrip(tmp_path):
    arr = np.random.rand(25, 8, 8).astype(np.float32)
    out_dir = tmp_path / "clip" / "encoding25_dt10"
    out_dir.mkdir(parents=True)
    np.save(out_dir / "000001.npy", arr)
    loaded = np.load(out_dir / "000001.npy").astype(np.float32)
    validate_encoding25_tensor(loaded)  # must not raise
    assert np.allclose(arr, loaded)
```

- [ ] Add Group 2 — Dataset `_load_encoded_flow_spike` actual loading tests (`@pytest.mark.integration`):

```python
# ---------------------------------------------------------------------------
# Group 2 — Dataset _load_encoded_flow_spike actual loading
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_load_encoded_flow_spike_returns_correct_shape(tmp_path):
    artifact_dir = tmp_path / "clip_a" / "encoding25_dt10"
    artifact_dir.mkdir(parents=True)
    arr = np.zeros((25, 8, 8), dtype=np.float32)
    np.save(artifact_dir / "000001.npy", arr)

    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds.spike_root = tmp_path
    ds.spike_flow_root = "auto"
    ds.spike_flow_dt = 10
    ds.filename_tmpl = "06d"

    result = ds._load_encoded_flow_spike("clip_a", 1)
    assert result.shape == (25, 8, 8)
    assert result.dtype == np.float32


@pytest.mark.integration
def test_load_encoded_flow_spike_auto_root_resolves_to_spike_root(tmp_path):
    artifact_dir = tmp_path / "clip_a" / "encoding25_dt10"
    artifact_dir.mkdir(parents=True)
    np.save(artifact_dir / "000001.npy", np.zeros((25, 8, 8), dtype=np.float32))

    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds.spike_root = tmp_path
    ds.spike_flow_root = "auto"
    ds.spike_flow_dt = 10
    ds.filename_tmpl = "06d"

    result = ds._load_encoded_flow_spike("clip_a", 1)
    assert result.shape == (25, 8, 8)


@pytest.mark.integration
def test_load_encoded_flow_spike_explicit_root_overrides_spike_root(tmp_path):
    explicit_root = tmp_path / "explicit"
    artifact_dir = explicit_root / "clip_a" / "encoding25_dt10"
    artifact_dir.mkdir(parents=True)
    np.save(artifact_dir / "000001.npy", np.zeros((25, 8, 8), dtype=np.float32))

    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds.spike_root = tmp_path / "other"
    ds.spike_flow_root = str(explicit_root)
    ds.spike_flow_dt = 10
    ds.filename_tmpl = "06d"

    result = ds._load_encoded_flow_spike("clip_a", 1)
    assert result.shape == (25, 8, 8)
```

- [ ] Add Group 3 — SCFlowWrapper forward output shapes tests (`@pytest.mark.integration`):

```python
# ---------------------------------------------------------------------------
# Group 3 — SCFlowWrapper forward output shapes
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_scflow_wrapper_forward_returns_4_scales():
    mock_model = _MockSCFlowModel()
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    wrapper.input_type = "spike"
    wrapper.device = torch.device("cpu")
    wrapper.dt = 10
    wrapper.model = mock_model

    spk1 = torch.randn(1, 25, 16, 16)
    spk2 = torch.randn(1, 25, 16, 16)
    result = wrapper.forward(spk1, spk2)
    assert len(result) == 4


@pytest.mark.integration
def test_scflow_wrapper_forward_output_shapes():
    mock_model = _MockSCFlowModel()
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    wrapper.input_type = "spike"
    wrapper.device = torch.device("cpu")
    wrapper.dt = 10
    wrapper.model = mock_model

    spk1 = torch.randn(1, 25, 16, 16)
    spk2 = torch.randn(1, 25, 16, 16)
    result = wrapper.forward(spk1, spk2)
    expected_shapes = [(1, 2, 16, 16), (1, 2, 8, 8), (1, 2, 4, 4), (1, 2, 2, 2)]
    for flow, expected in zip(result, expected_shapes):
        assert tuple(flow.shape) == expected


@pytest.mark.integration
def test_scflow_wrapper_forward_passes_dt_to_model():
    mock_model = _MockSCFlowModel()
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    wrapper.input_type = "spike"
    wrapper.device = torch.device("cpu")
    wrapper.dt = 10
    wrapper.model = mock_model

    spk1 = torch.randn(1, 25, 16, 16)
    spk2 = torch.randn(1, 25, 16, 16)
    wrapper.forward(spk1, spk2)
    assert mock_model.received_dt == 10
```

- [ ] Add Group 4 — ModelPlain `netG_forward` routing tests (`@pytest.mark.integration`):

```python
# ---------------------------------------------------------------------------
# Group 4 — ModelPlain netG_forward routing
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_netg_forward_passes_flow_spike_for_scflow():
    net = _RecordingNet()
    model = _build_stub_plain_model_with_net("scflow", net)
    data = {
        "L": torch.randn(1, 4, 7, 8, 8),
        "L_flow_spike": torch.randn(1, 4, 25, 8, 8),
    }
    model.feed_data(data, need_H=False)
    model.netG_forward()
    assert "flow_spike" in net.last_call_kwargs
    assert tuple(net.last_call_kwargs["flow_spike"].shape) == (1, 4, 25, 8, 8)


@pytest.mark.integration
def test_netg_forward_omits_flow_spike_for_non_scflow():
    net = _RecordingNet()
    model = _build_stub_plain_model_with_net("spynet", net)
    data = {
        "L": torch.randn(1, 4, 7, 8, 8),
    }
    model.feed_data(data, need_H=False)
    model.netG_forward()
    assert "flow_spike" not in net.last_call_kwargs
```

- [ ] Add Group 5 — VRT `get_flow_2frames` complete output tests (`@pytest.mark.integration`):

```python
# ---------------------------------------------------------------------------
# Group 5 — VRT get_flow_2frames complete output
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_vrt_get_flow_2frames_backward_forward_count():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    flow_spike = torch.randn(1, 4, 25, 16, 16)
    flows_backward, flows_forward = vrt.get_flow_2frames(x, flow_spike=flow_spike)
    assert len(flows_backward) == 4
    assert len(flows_forward) == 4


@pytest.mark.integration
def test_vrt_get_flow_2frames_flow_shape():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    flow_spike = torch.randn(1, 4, 25, 16, 16)
    flows_backward, _ = vrt.get_flow_2frames(x, flow_spike=flow_spike)
    # b=1, n-1=3, 2 channels, h=16, w=16 at scale i=0 → shape (1, 3, 2, 16, 16)
    # flows_backward has 4 elements (one per scale)
    assert len(flows_backward) == 4
    assert tuple(flows_backward[0].shape) == (1, 3, 2, 16, 16)


@pytest.mark.integration
def test_vrt_get_flow_2frames_uses_flow_spike_not_x():
    vrt = VRT.__new__(VRT)
    spy = _DummySpikeFlow()
    vrt.spynet = spy
    x = torch.randn(1, 4, 7, 16, 16)
    flow_spike = torch.randn(1, 4, 25, 16, 16)
    vrt.get_flow_2frames(x, flow_spike=flow_spike)
    # spynet received 25-channel spike frames, not 7-channel x frames
    assert spy.calls[0][0].shape[1] == 25
```

