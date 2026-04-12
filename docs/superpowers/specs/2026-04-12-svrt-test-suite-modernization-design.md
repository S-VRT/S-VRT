# S-VRT Test Suite Modernization Design (Plan B)

## Context
Current `tests/` already has broad coverage, but category boundaries and execution contracts are inconsistent: some legacy tests are script-like, markers are uneven, and data-dependent checks are mixed with local-fast checks. The project now has updated dual-input/fusion behavior and server-side dataset paths in `options/gopro_rgbspike_server.json`, so the test system needs restructuring without disrupting core model code.

## Goal
Upgrade the test system end-to-end with full-layer coverage:
- Unit tests
- Integration tests
- Smoke tests
- E2E tests (server dataset aware)

while preserving existing validated behavior and making old tests compatible with current project state.

## Scope
In scope:
- Refactor tests and test infra under `tests/`
- Add marker governance and execution entrypoints
- Add server-data-aware E2E tests based on `options/gopro_rgbspike_server.json`
- Modernize outdated tests to current behavior/contracts

Out of scope:
- Changing core training/model behavior for test convenience
- Large architecture changes outside test code

## Architecture
### 1) Layered test taxonomy
- `unit`: isolated behavior, no real dataset dependency.
- `integration`: module collaboration contracts (dataset->model input path, fusion adapter flow, optical flow wiring).
- `smoke`: quick startup/forward sanity checks for critical runtime paths.
- `e2e`: minimal real-data closed loop on compute platform, server-option driven.

### 2) Unified test infrastructure
- Extend `tests/conftest.py` with:
  - fixture to load `options/gopro_rgbspike_server.json`
  - helpers for data path validation
  - standardized conditional skip helpers (`missing_dataset`, `missing_cuda`, `missing_optional_dep`)
- Register and normalize markers in `pytest.ini` (or equivalent pytest config):
  - `unit`, `integration`, `smoke`, `e2e`, `slow`

### 3) Legacy test modernization policy
- Keep existing file locations where practical; avoid unnecessary renames.
- Convert script-style tests (`print`, `main` execution branches) into assertion-driven pytest tests.
- Replace fragile hard-coded personal paths with config/fixture-driven values.
- Align stale assertions with current dual-input/fusion contracts.

### 4) E2E design
New file: `tests/e2e/test_gopro_rgbspike_server_e2e.py`
- Read server options from `options/gopro_rgbspike_server.json`
- Validate GT/LQ/Spike paths exist on compute platform
- Build minimal dataset/model runtime path
- Run one minimal batch forward (or one tiny step)
- Assert output tensor contract and no runtime contract violation
- If environment/data unavailable: explicit `pytest.skip(reason=...)`

## Data Flow
1. Local dev/CI primarily executes `unit + integration + smoke`.
2. Compute platform runs `e2e` against real server paths.
3. Results are category-scoped, reducing noise and improving triage.

## Error Handling & Stability Rules
- Data-dependent tests must never fail due to absent environment without a clear skip reason.
- GPU-only checks guarded by `skipif` conditions.
- Optional dependency checks produce explicit skip, not opaque import failures.
- Preserve deterministic seeds for sensitive integration paths where feasible.

## Implementation Plan Inputs
### Files to modify
- `tests/conftest.py`
- `pytest.ini` (or equivalent config used by this repo)
- `tests/run_tests.py`
- `tests/README.md`
- selected legacy tests under:
  - `tests/models/`
  - `tests/smoke/`
  - `tests/op/`

### Files to add
- `tests/e2e/test_gopro_rgbspike_server_e2e.py`

## Verification Matrix
1. `pytest -m unit -q`
2. `pytest -m integration -q`
3. `pytest -m smoke -q`
4. `pytest -m "unit or integration or smoke" -q`
5. `pytest -m e2e -q` (compute platform)

Acceptance expectations:
- Unit/integration/smoke stable in normal dev environment
- E2E executes on compute platform when data available
- E2E emits explicit skip reason when prerequisites missing

## Risks and Mitigations
- Risk: legacy tests break due to hidden assumptions.
  - Mitigation: incremental refactor with category-specific runs after each batch.
- Risk: E2E too slow.
  - Mitigation: strictly minimal batch/step path.
- Risk: marker drift over time.
  - Mitigation: document marker policy in `tests/README.md` and enforce via `run_tests.py` presets.

## Deliverables
1. Refactored tests with stable categorization and modern assertions.
2. Server-option aware E2E test.
3. Updated test execution docs and scripts.
4. Reproducible verification commands by test layer.
