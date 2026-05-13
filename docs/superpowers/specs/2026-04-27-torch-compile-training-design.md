# Torch Compile Training Wrapper Design

## Goal

Add optional `torch.compile` support to the S-VRT training path without changing VRT network architecture, attention blocks, fusion algorithms, optical-flow modules, or KAIR-comparable model behavior.

The optimization must live at the training wrapper level. KAIR's original VRT remains the comparison baseline for model and algorithm semantics.

## Non-Goals

- Do not edit KAIR source files.
- Do not rewrite VRT stages, attention, DCNv4, optical-flow, or fusion operator algorithms for compile compatibility.
- Do not enable compile by default for all experiments.
- Do not compile EMA `netE` in the first implementation.
- Do not introduce FSDP/ZeRO in this change.

## Configuration

Add an optional `train.compile` dictionary:

```json
"compile": {
  "enable": false,
  "mode": "default",
  "fullgraph": false,
  "dynamic": true,
  "backend": "inductor",
  "fallback_on_error": true
}
```

Defaults are conservative:

- `enable=false` preserves current eager behavior and experiment comparability.
- `fullgraph=false` tolerates graph breaks from checkpointing, custom CUDA ops, LoRA/fusion branches, and validation-only paths.
- `dynamic=true` allows phase changes and variable input sizes.
- `fallback_on_error=true` logs a warning and continues in eager mode if compile fails during wrapping.

Server experiment configs may opt in explicitly with `mode="reduce-overhead"` after smoke validation.

## Architecture

Compile is added in `ModelBase.model_to_device()` as a wrapper-level step:

1. Move the bare network to the selected device.
2. Optionally call `torch.compile(network, ...)`.
3. Wrap the result with `DistributedDataParallel` or `DataParallel`.
4. Apply existing DDP static-graph behavior after DDP wrapping.

This keeps optimizer construction, DDP reducer setup, checkpoint load/save, and parameter freezing close to the current flow.

`get_bare_model()` must understand compiled modules well enough for save/load and phase transitions. If PyTorch wraps compiled modules with an `_orig_mod` attribute, that attribute should be unwrapped after DDP/DataParallel unwrapping.

## Data Flow

Training construction remains:

`define_G(opt)` -> `model_to_device(netG)` -> `init_train()` -> `load()` -> LoRA/freeze/loss/optimizer/scheduler setup.

Compile happens inside `model_to_device()` before DDP/DataParallel wrapping. No tensor shapes, model inputs, targets, losses, checkpoint keys, or KAIR-compatible state dict names should change.

When phase 2 re-wraps DDP in `ModelVRT.optimize_parameters()`, it should continue calling `model_to_device(self.get_bare_model(self.netG))`, so compile behavior is consistent across initial setup and re-wrap.

## Error Handling

If `train.compile.enable=true` and `torch.compile` is unavailable or raises during wrapping:

- With `fallback_on_error=true`, rank 0 logs a warning and training continues in eager mode.
- With `fallback_on_error=false`, the original exception is raised.

Compile must never silently alter checkpoint state dict keys. Save/load should operate on the original model module when possible.

## Testing

Unit tests should cover:

- `model_to_device()` calls `torch.compile` before DDP/DataParallel when `train.compile.enable=true`.
- Compile is skipped when config is absent or `enable=false`.
- Compile failure falls back to eager when `fallback_on_error=true`.
- Compile failure raises when `fallback_on_error=false`.
- `get_bare_model()` unwraps DDP/DataParallel and compiled `_orig_mod` wrappers for checkpoint-compatible state dict access.

No model-algorithm tests should require changes, because this feature is a wrapper-level training optimization.

## Rollout

1. Implement the wrapper-level compile helper and tests.
2. Keep all existing configs default eager.
3. Add compile config to server configs with `enable=false` as documented knobs.
4. Run targeted unit tests.
5. Run a short smoke train with compile enabled before using it for long training.
