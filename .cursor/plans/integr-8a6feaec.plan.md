<!-- 8a6feaec-2784-4023-a966-a96d205ab1a2 bc19998a-4b77-42fb-b64d-1374d945c16f -->
# Integrate SEA-RAFT as pluggable optical-flow module

## Goal

Add SEA-RAFT as an alternative optical-flow backend to the project so that experiments can switch between `spynet` and `sea_raft` via `@options` JSON. Copy the necessary source from the `SEA-RAFT-main` folder into the repo (adapted), include a default checkpoint in the repository, expose a small, stable API both implementations implement, and add smoke unit tests.

## High-level approach

- Define a small Optical Flow adapter API that both `SpyNet` and `SEA-RAFT` will implement. This makes them interchangeable for ablation/experiments.
- Add a package `models/optical_flow/` with:
                                - `base.py` — abstract base class / interface and helper utilities.
                                - `spynet_wrapper.py` — thin adapter that wraps existing SpyNet usage into the new API (reused code).
                                - `sea_raft.py` — vendor-pasted and adapted SEA-RAFT implementation that implements the same API.
                                - `__init__.py` — factory `create_optical_flow(module_name, checkpoint, device, **kwargs)`.
- Copy the minimal set of SEA-RAFT source files from `SEA-RAFT-main` into `models/optical_flow/_vendor/sea_raft/`, update internal imports to be relative and remove any dependency on the original folder. After verification, delete `SEA-RAFT-main` from repository.
- Add default checkpoints under `weights/optical_flow/sea_raft.ckpt` and ensure `weights/optical_flow/spynet.ckpt` already exists (you said spynet checkpoint is present). Update `.gitignore` / LFS notes if needed.
- Add options schema in `@options` JSON files: new `optical_flow` section controls which backend and which checkpoint to use.
- Add a smoke test `tests/models/test_optical_flow_smoke.py` that constructs the module for both `spynet` and `sea_raft`, runs a forward with a tiny tensor (e.g., 2 frames 3x64x64) and asserts output shape and numeric finiteness.

## Proposed API (small and stable)

Make the adapters implement this interface in `models/optical_flow/base.py`:

```python
class OpticalFlowModule(torch.nn.Module):
    """Adapter base class for optical flow backends.

    Required methods:
  - forward(frame1: Tensor, frame2: Tensor) -> Tensor  # returns flow with shape (B, 2, H, W)
  - load_checkpoint(path: str) -> None
    """
    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def load_checkpoint(self, path: str) -> None:
        raise NotImplementedError
```

Rationale: the project currently calls a flow module to estimate flows between frames. A simple pairwise forward signature keeps integration minimal and supports both per-pair and multi-scale flows inside the implementation.

Adapters should also accept `device` and `eval_mode` on construction.

## Files to add / edit (concrete)

- Add: `models/optical_flow/base.py` (new)
- Add: `models/optical_flow/__init__.py` (new) — factory `create_optical_flow(...)`
- Add or update: `models/optical_flow/spynet_wrapper.py` (new or thin wrapper) — reuse existing SpyNet loader and expose the `OpticalFlowModule` API
- Add: `models/optical_flow/sea_raft.py` (new) — vendor-pasted SEA-RAFT code adapted into a single module plus a small adapter class implementing `OpticalFlowModule`
- Add: `models/optical_flow/_vendor/sea_raft/*` — copied SEA-RAFT source files required by `sea_raft.py` (only for development; will be removed after adaptation; final integrated code in `sea_raft.py` must not import from `SEA-RAFT-main`)
- Edit: places where SpyNet is currently instantiated/loaded (likely in `models/model_vrt.py` or in a dedicated loader). Replace direct SpyNet construction with `from models.optical_flow import create_optical_flow` and call the factory using options.
- Edit: `options/*.json` (e.g., `options/gopro_rgbspike_local_debug.json`) — add new `optical_flow` section

Example `@options` JSON fragment:

```json
"optical_flow": {
  "module": "sea_raft",           
  "checkpoint": "weights/optical_flow/sea_raft.ckpt",
  "device": "cuda:0",
  "params": {}
}
```

## Tests

- Add `tests/models/test_optical_flow_smoke.py` that:
                                - Builds a tiny tensor batch: two frames with shape `(B=1, C=3, H=64, W=64)`
                                - Instantiates both `spynet` and `sea_raft` via factory `create_optical_flow` using the default checkpoints in `weights/optical_flow/`
                                - Calls `forward` and asserts output shape `(1, 2, 64, 64)` and that values are finite
                                - Runs in CPU-only mode on CI to avoid GPU requirement (use a very small model or ensure weights in repo are CPU-compatible); if SEA-RAFT checkpoint is GPU-only, fallback to randomly initialized model for smoke test but still validate shapes.

Note: smoke tests should be fast and not download anything.

## Vendor copy strategy and cleanup

1. Copy minimal source files from `SEA-RAFT-main/` into `models/optical_flow/_vendor/sea_raft/` as a temporary staging area for adaptation. Update imports to be local relative imports.
2. Consolidate the necessary functions/classes into `models/optical_flow/sea_raft.py` (so runtime code imports only from `models.optical_flow` package). The `sea_raft.py` file will contain the user-facing adapter class plus any helper classes extracted from vendor files.
3. After verification and tests passing, remove `SEA-RAFT-main` and the `_vendor` staging folder, leaving only `sea_raft.py` and weights under `weights/optical_flow/`.

## Config changes

- Add `optical_flow` top-level key to project option JSON files. Update any config schema utilities if present.
- Ensure `create_optical_flow` reads `module` and `checkpoint` and applies `device`.

## Risks & approvals

- License: You must confirm the SEA-RAFT license permits copying source into this repo. If it requires attribution or has incompatible terms, we must follow them (e.g., keep LICENSE and attribution files in repo). Please confirm.
- Checkpoint size: adding SEA-RAFT checkpoint to repo increases repo size; consider Git LFS.

## Execution order (summary)

1. Add `models/optical_flow` API + `spynet` adapter.
2. Add `sea_raft` adapter by vendor-copying SEA-RAFT code into `_vendor/` and adapting into `sea_raft.py`.
3. Add default checkpoints under `weights/optical_flow/`.
4. Update configuration JSON examples.
5. Add smoke tests and run locally.
6. Remove `SEA-RAFT-main` and any temporary vendor folders, leaving only adapted code.

---

If this plan looks good I will convert it to a todo list and start implementing the first task (creating the `models/optical_flow` API and `spynet` wrapper). If you want to change the API signature or the checkpoints strategy, tell me now.

### To-dos

- [ ] Create `models/optical_flow/base.py` and `models/optical_flow/__init__.py` with factory API
- [ ] Add `models/optical_flow/spynet_wrapper.py` adapter that implements `OpticalFlowModule` and reuses existing SpyNet code
- [ ] Copy necessary SEA-RAFT source from `SEA-RAFT-main` into `models/optical_flow/_vendor/sea_raft` and adapt imports
- [ ] Create `models/optical_flow/sea_raft.py` consolidating vendor code and exposing adapter implementing `OpticalFlowModule`
- [ ] Add default checkpoints under `weights/optical_flow/` and update repo LFS or instructions
- [ ] Update `options/*.json` and replace direct SpyNet instantiation with `create_optical_flow` factory
- [ ] Add `tests/models/test_optical_flow_smoke.py` to validate both `spynet` and `sea_raft` forward pass
- [ ] Remove `SEA-RAFT-main` and temporary `_vendor` once final `sea_raft.py` is self-contained