# VRT/S-VRT Test Suite

This directory contains the modernized layered test suite for VRT/S-VRT.

## Test Layers

- `unit`: Fast isolated tests.
- `integration`: Cross-module integration contracts.
- `smoke`: Quick end-to-end sanity checks for key paths.
- `e2e`: Real platform/data end-to-end coverage (may skip when prerequisites are missing).

## Runner Commands

```bash
python tests/run_tests.py --unit
python tests/run_tests.py --integration
python tests/run_tests.py --smoke
python tests/run_tests.py --e2e
```

## Pytest Marker Commands

```bash
python -m pytest -m "unit or integration or smoke" -q
python -m pytest -m e2e -q
```

## Notes

- `--smoke` and `--e2e` are marker-based dispatches.
- E2E tests read server options from `options/gopro_rgbspike_server.json`.
- E2E tests should PASS on compute platform with valid data paths, or SKIP with explicit reasons when prerequisites are unavailable.
