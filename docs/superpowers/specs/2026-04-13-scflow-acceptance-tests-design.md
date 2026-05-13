# SCFlow Acceptance Test Suite Design

**Date:** 2026-04-13
**Scope:** Acceptance tests for the SCFlow strict semantic integration in S-VRT.
**Goal:** Verify that all strict 25-slice spike semantics are correctly enforced and that the full data flow (encoding25 → dataset → model → VRT → SCFlow wrapper) works correctly with synthetic data.

---

## Test File Structure

```
tests/models/
  test_optical_flow_scflow_contract.py    # Layer 1: contract/unit tests (existing + additions)
  test_optical_flow_scflow_integration.py # Layer 2: functional integration tests (new)
```

Both files run in CI without real dataset paths or model weights.

---

## Layer 1: Contract Tests (existing file, additions)

File: `tests/models/test_optical_flow_scflow_contract.py`

All tests are pure unit tests: no file I/O, no real models, millisecond-level runtime.
Mark with `@pytest.mark.unit`.

### Group A — encoding25 utility contracts

| Test | What it verifies |
|------|-----------------|
| `test_validate_encoding25_tensor_rejects_2d_input` | ndim=2 raises ValueError |
| `test_validate_encoding25_tensor_rejects_4d_input` | ndim=4 raises ValueError |
| `test_validate_encoding25_tensor_accepts_valid` | [25,H,W] passes without error |
| `test_build_output_dir_rejects_zero_dt` | dt=0 raises ValueError |
| `test_build_output_dir_rejects_negative_dt` | dt=-1 raises ValueError |
| `test_compute_center_index_formula` | center_offset + (frame_index - clip_start_frame) * dt |
| `test_validate_center_bounds_rejects_left_boundary` | center - 12 < edge_margin raises ValueError |
| `test_validate_center_bounds_rejects_right_boundary` | center + 12 >= total_length - edge_margin raises ValueError |
| `test_validate_center_bounds_accepts_valid_center` | valid center passes without error |
| `test_build_centered_window_rejects_2d_spike_matrix` | ndim=2 raises ValueError |
| `test_build_centered_window_rejects_length_not_25` | length=24 raises ValueError |
| `test_build_centered_window_rejects_center_too_close_to_start` | center=5 raises ValueError |
| `test_build_centered_window_rejects_center_too_close_to_end` | center near end raises ValueError |

### Group B — SCFlowWrapper contracts

| Test | What it verifies |
|------|-----------------|
| `test_scflow_wrapper_rejects_3d_input` | spk1 ndim=3 raises ValueError |
| `test_scflow_wrapper_rejects_5d_input` | spk1 ndim=5 raises ValueError |
| `test_scflow_wrapper_rejects_wrong_channels_on_spk2` | spk1 correct, spk2 channels=11 raises ValueError |

### Group C — ModelPlain contracts

| Test | What it verifies |
|------|-----------------|
| `test_model_plain_rejects_l_flow_spike_ndim_4` | L_flow_spike ndim=4 raises ValueError |
| `test_model_plain_rejects_l_flow_spike_wrong_channels` | L_flow_spike channels=11 raises ValueError |
| `test_model_plain_stores_l_flow_spike_on_valid_input` | self.L_flow_spike set correctly |
| `test_model_plain_flow_module_alias_spike_flow` | `spike_flow` module name maps to `scflow` |
| `test_model_plain_clears_l_flow_spike_for_non_scflow` | non-scflow module sets L_flow_spike=None |

### Group D — VRT contracts

| Test | What it verifies |
|------|-----------------|
| `test_vrt_rejects_flow_spike_ndim_4` | flow_spike ndim=4 raises ValueError |
| `test_vrt_rejects_flow_spike_batch_mismatch` | flow_spike B≠x.B raises ValueError |
| `test_vrt_rejects_flow_spike_time_mismatch` | flow_spike T≠x.T raises ValueError |
| `test_vrt_rejects_flow_spike_spatial_mismatch` | flow_spike H/W≠x.H/W raises ValueError |
| `test_vrt_rejects_flow_spike_wrong_channels` | flow_spike channels=11 raises ValueError |

### Group E — Dataset contracts

| Test | What it verifies |
|------|-----------------|
| `test_dataset_no_spike_flow_config_disables_encoding25` | no spike_flow key → use_encoding25_flow=False |
| `test_dataset_load_path_construction_auto_mode` | auto root → path uses self.spike_root |
| `test_dataset_load_path_construction_explicit_root` | explicit root → path uses that root |

---

## Layer 2: Functional Integration Tests (new file)

File: `tests/models/test_optical_flow_scflow_integration.py`

All tests use synthetic data and mock/stub objects. No real weights or dataset paths required.
Mark with `@pytest.mark.integration`.

### Group 1 — encoding25 data round-trip

| Test | What it verifies |
|------|-----------------|
| `test_build_centered_window_extracts_correct_slice` | window = spike_matrix[center-12:center+13], values match |
| `test_compute_center_and_build_window_pipeline` | compute_center_index → build_centered_window produces correct 25-frame window |
| `test_encoding25_npy_roundtrip` | write [25,8,8] to .npy, load back, passes validate_encoding25_tensor, values preserved |

### Group 2 — Dataset `_load_encoded_flow_spike` actual loading

| Test | What it verifies |
|------|-----------------|
| `test_load_encoded_flow_spike_returns_correct_shape` | tmp_path artifact → shape [25,8,8], dtype float32 |
| `test_load_encoded_flow_spike_auto_root_resolves_to_spike_root` | spike_flow_root="auto" uses self.spike_root |
| `test_load_encoded_flow_spike_explicit_root_overrides_spike_root` | explicit root path used instead of spike_root |

### Group 3 — SCFlowWrapper forward output shapes

Uses a mock SCFlow model injected into the wrapper (bypasses real weights).

| Test | What it verifies |
|------|-----------------|
| `test_scflow_wrapper_forward_returns_4_scales` | output list length == 4 |
| `test_scflow_wrapper_forward_output_shapes` | shapes: [1,2,16,16], [1,2,8,8], [1,2,4,4], [1,2,2,2] |
| `test_scflow_wrapper_forward_passes_dt_to_model` | mock model receives dt=self.dt kwarg |

### Group 4 — ModelPlain `netG_forward` routing

Uses a stub netG that records call arguments.

| Test | What it verifies |
|------|-----------------|
| `test_netg_forward_passes_flow_spike_for_scflow` | netG called with flow_spike= matching L_flow_spike tensor |
| `test_netg_forward_omits_flow_spike_for_non_scflow` | netG called without flow_spike kwarg |

### Group 5 — VRT `get_flow_2frames` complete output

Uses `_DummySpikeFlow` (returns 4-scale fake flows).

| Test | What it verifies |
|------|-----------------|
| `test_vrt_get_flow_2frames_backward_forward_count` | x=[1,4,7,16,16] → backward len=3, forward len=3 |
| `test_vrt_get_flow_2frames_flow_shape` | each flow tensor shape=[1,2,16,16] |
| `test_vrt_get_flow_2frames_uses_flow_spike_not_x` | mock spynet receives flow_spike frames, not x frames |

---

## Pytest Markers

Both files use existing markers from `pytest.ini`:
- Layer 1: `@pytest.mark.unit`
- Layer 2: `@pytest.mark.integration`

Run commands:
```bash
# Layer 1 only
.venv/bin/python -m pytest tests/models/test_optical_flow_scflow_contract.py -v

# Layer 2 only
.venv/bin/python -m pytest tests/models/test_optical_flow_scflow_integration.py -v

# Both layers
.venv/bin/python -m pytest tests/models/test_optical_flow_scflow_contract.py tests/models/test_optical_flow_scflow_integration.py -v
```

---

## Constraints

- No real dataset paths, no real model weights, no network I/O
- All tests must pass on a machine with only the project's Python dependencies installed
- `tmp_path` fixture used for all file system operations
- Mock/stub objects injected via `__new__` + attribute assignment (consistent with existing test style)
- `_DummySpikeFlow` and other shared stubs defined in the integration test file directly (not imported from the contract test file, to keep files independent)
- SCFlowWrapper mock: replace `wrapper.model` attribute after `__new__` construction, do not subclass
- Total runtime target: < 10 seconds for both layers combined
