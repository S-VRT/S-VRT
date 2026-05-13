# S-VRT SCFlow Strict Semantic Integration Design

> **Update (2026-04-16):** 本 spec 的 encoding25 契约已被子帧扩展（subframe encoding25）所扩展。
> 当 early fusion 启用时，`L_flow_spike` 的时间维度从 N 变为 N×S（S = `spike_flow.subframes`），
> artifact 目录变为 `encoding25_dt{dt}_s{S}`。全局 `center_offset + k × dt` 索引方案已废弃，
> 改为基于每个 `.dat` 自身 T_raw 的局部子中心选取。
> 详见 `docs/superpowers/specs/2026-04-16-scflow-subframe-encoding25-design.md`。

## 1. Background

Current SCFlow integration in S-VRT is contract-incomplete:
- SCFlow expects spike-sequence input with 25 temporal slices per frame pair.
- VRT currently routes full restoration tensor channels to SCFlow when `input_type='spike'`.
- Existing RGB+Spike reconstruction input (`spikecv_tfp`, `in_chans=11`) does not preserve SCFlow's original 25-step temporal semantics.

Goal: integrate SCFlow in a strict-semantic way so optical-flow estimation uses true spike temporal resolution, while restoration path remains backward-compatible.

## 2. Goals and Non-Goals

### Goals
- Preserve SCFlow optical-flow semantics: flow input shape `[B, T, 25, H, W]`.
- Keep restoration input pipeline (`L`) unchanged unless explicitly configured.
- Add explicit, fail-fast contracts across config, dataset, and model.
- Provide an offline pre-encoding workflow compatible with current dataset layout.
- Add tests that lock the SCFlow contract and prevent silent regressions.

### Non-Goals
- No immediate LMDB migration for encoded spike flow.
- No approximation adapters from TFP bins to pseudo-25 slices.
- No redesign of SCFlow architecture itself in this phase.

## 3. Recommended Storage and Module Placement

### 3.1 Encoding logic module
- New module: `data/spike_recc/encoding25.py`
- Responsibilities:
  - Build centered 25-slice windows from raw spike matrix.
  - Enforce `length=25` and configurable `dt` spacing.
  - Provide reusable path helpers (e.g., `build_output_dir`) for both tests and scripts.
  - Deterministic index policy for train/test.

### 3.2 Offline preparation script
- New script: `scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py`
- Responsibilities:
  - Traverse spike dataset folders.
  - Generate `encoding25_dt{dt}` `.npy` files.
  - Support dry-run and overwrite guards.
  - Emit summary stats and missing-file report.

### 3.3 Artifact placement
Use dataset-local artifacts (recommended phase-1):
- Input spike file: `<dataroot_spike>/<clip>/spike/<frame>.dat`
- Encoded output: `<dataroot_spike>/<clip>/encoding25_dt{dt}/<frame>.npy`

Rationale:
- Closest to original SCFlow workflow.
- Minimal path indirection for immediate integration.
- Easier diagnostics during server runs.

## 4. Configuration Contract

Add a dedicated optical-flow spike config namespace under each split:
- `datasets.{train|test}.spike_flow.representation = "encoding25"`
- `datasets.{train|test}.spike_flow.dt = 10` (or 20)
- `datasets.{train|test}.spike_flow.root = "auto"` (default follows `dataroot_spike`)

Hard contract rules:
- If `netG.optical_flow.module == "scflow"`, then:
  - `spike_flow.representation` must be `encoding25`.
  - dataset must provide `L_flow_spike` with channel size 25.
- Violations must raise explicit `ValueError` with actionable hints.
Temporal alignment contract (strict semantic):
- Let `f` be current frame index parsed from frame filename (`000123` -> `f=123`).
- Let `f0` be clip start frame from metadata (`start_frame` in meta info line).
- Let `k = f - f0` be clip-local frame index.
- Define center slice index:
  - `center = center_offset + k * dt`
- Fixed defaults for phase-1 compatibility:
  - `center_offset = 40`
  - `edge_margin = 40`
  - `window_length = 25` (`half=12`)
- Validity condition:
  - `center - 12 >= edge_margin` and `center + 12 < T - edge_margin`
  - where `T` is total spike time length in raw matrix.
- If invalid, fail fast with explicit message containing `clip`, `frame`, `center`, `T`, and configured parameters.

## 5. Data Pipeline Changes

Extend `TrainDatasetRGBSpike` and test counterpart to optionally output:
- `L` (existing restoration input; unchanged)
- `L_flow_spike` (new flow-only input), shape `[T, 25, H, W]`

Loading behavior:
- For SCFlow mode, load pre-encoded `.npy` from `encoding25_dt{dt}`.
- Resize spatially to match RGB crop / frame size.
- Keep dtype/normalization simple and deterministic (float32, no ImageNet normalization).
- Ensure encoded filename matches RGB frame index so train/test use identical temporal alignment policy.

Failure behavior:
- Missing encoded file: fail with explicit path and regeneration hint.
- Wrong channel count: fail with expected vs actual count.

## 6. Model/VRT Integration Changes

### 6.1 Model ingress
In `ModelPlain` input assembly:
- Continue building `self.L` from existing concat/dual flow.
- When `module=scflow`, require and store `self.L_flow_spike`.
- Pass both tensors to network call (`netG(self.L, flow_spike=self.L_flow_spike)`) or equivalent agreed signature.

### 6.2 VRT forward and flow path
- Extend VRT forward to accept optional flow-specific spike tensor.
- In `get_flow_2frames`:
  - If backend is SCFlow, consume only `flow_spike` tensor.
  - Do not use `x_flow = x` for SCFlow branch.
- Keep RGB-flow backends (`spynet`, `sea_raft`) unchanged.

### 6.3 SCFlow wrapper contract hardening
- Add strict input validation in wrapper:
  - ndim == 4, channels == 25.
- Error message should include received shape and expected format.

## 7. Tests

### 7.1 New tests
- `tests/models/test_optical_flow_scflow_contract.py`
  - SCFlow wrapper rejects non-25 channel inputs.
  - VRT SCFlow branch requires `L_flow_spike`.
  - Correct `L_flow_spike` shape passes contract checks.

### 7.2 Update existing smoke philosophy
- Keep existing `spynet/sea_raft` smoke tests.
- Add SCFlow contract-only tests that do not require pretrained checkpoints.

## 8. Execution and Verification Plan (Server)

Because local environment cannot run torch stack, verification will be executed on server via `cybertron-server`:
1. Sync branch to server workspace.
2. Run encoding preparation script for selected split.
3. Run contract-focused tests.
4. Run one minimal forward/integration smoke with `module=scflow` and strict config.

Required evidence:
- Encoding generation summary log.
- Passing contract test outputs.
- One successful SCFlow integration smoke output (shape logs).

## 9. Risks and Mitigations

- Risk: Encoding generation cost/time on large datasets.
  - Mitigation: dry-run, incremental generation, and skip-existing behavior.

- Risk: Signature changes to VRT/model call sites.
  - Mitigation: optional args with strict gating only when `module=scflow`.

- Risk: Path inconsistencies across local/server.
  - Mitigation: explicit `spike_flow.root` override and standardized diagnostics.

## 10. Rollout Sequence

1. Add encoding module + script.
2. Add dataset support for `L_flow_spike`.
3. Add model/VRT strict SCFlow routing.
4. Add SCFlow contract tests.
5. Server-side verification and iterate.

## 11. Acceptance Criteria

- SCFlow path no longer consumes restoration `L` channels directly.
- SCFlow receives only `[B, T, 25, H, W]` derived from offline encoding.
- Misconfiguration fails fast with clear error messages.
- Added tests catch contract regressions pre-runtime.
- Server verification artifacts confirm end-to-end strict-semantic path works.

