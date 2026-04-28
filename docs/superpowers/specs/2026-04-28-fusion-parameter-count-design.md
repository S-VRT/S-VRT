# Fusion Parameter Count Design

## Goal

Add a reliable way to report the full parameter count of the configured fusion module for paper-scale reporting.

## Scope

The reported number is the complete fusion module size, not an operator-only number. Auxiliary layers owned by the fusion path, such as adapter-side spike upsampling/refinement layers, are part of the model and must be counted.

## Architecture

The implementation will add a reusable analysis helper and a small CLI under `scripts/analysis`. The CLI reads an existing option JSON, builds `netG` through the project's normal `utils_option.parse(..., is_train=False)` and `models.select_network.define_G()` path, and then counts parameters owned by fusion modules on the resulting VRT instance.

The primary module boundary is `netG.fusion_adapter`, because it owns the full runtime fusion path. If a future model exposes a fusion module differently, the helper can fail clearly instead of silently reporting zero. Parameters are counted by object identity so shared parameters are included once.

## Output

The CLI prints one complete count:

```text
Fusion parameters: 1,234,567 (1.235 M)
```

It may include config metadata, but it must not split the result into operator-only and adapter-only columns.

## Error Handling

If the option does not enable `input.strategy=fusion`, or the built model has no usable fusion module, the tool exits with a clear error. This prevents accidentally using a zero count for paper reporting.

## Testing

Tests cover the reusable counting helper with dummy modules:

- complete nested fusion subtrees are counted;
- shared parameters are deduplicated;
- missing or disabled fusion raises a clear error.

The tests do not depend on CUDA, mamba runtime execution, or real dataset paths.
