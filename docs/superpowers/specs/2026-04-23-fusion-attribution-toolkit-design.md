# Fusion Attribution Toolkit Design

## Purpose

S-VRT needs a reusable visualization and attribution tool for comparing fusion operators. The tool should support the current gated fusion design and future fusion variants such as concat, Mamba, PASE, or other operators without coupling analysis to the main training loop.

The tool is not intended to prove performance by heatmap alone. Its role is to localize restoration-sensitive regions affected by fusion. Quantitative ablations, error maps, and restoration metrics remain the primary evidence for performance claims.

## Goals

- Provide one offline analysis workflow for all S-VRT fusion variants.
- Keep attribution fully decoupled from main training, losses, optimizers, and phase-specific training logic.
- Reuse existing S-VRT model construction, checkpoint loading, and validation/test data loading.
- Export reproducible artifacts for paper figures and debugging.
- Support generic fusion attribution for every operator and optional operator-specific maps when an operator exposes extra internals.

## Non-Goals

- Do not add attribution logic to the training loop.
- Do not require every fusion operator to expose gate-like internals.
- Do not treat heatmaps as standalone proof that a fusion method improves restoration.
- Do not implement SCFlow or DCNv4 visualization in this tool; those require separate flow/alignment diagnostics.

## Command-Line Interface

The first implementation should add an offline entry point:

```bash
python scripts/analysis/fusion_attribution.py \
  --opt options/gopro_rgbspike_server.json \
  --checkpoint experiments/xxx/models/latest_G.pth \
  --samples docs/analysis/fusion_samples.json \
  --out results/fusion_attribution/gated_run \
  --cam-method gradcam \
  --target masked_charbonnier \
  --max-samples 8
```

Optional arguments:

```text
--baseline-opt
--baseline-checkpoint
--device cuda:0
--center-frame-only
--save-raw
--save-panel
--perturb-spike zero|shuffle|noise|temporal-drop
--mask-source manual|motion|error-topk
```

The first version should support manual masks, a single checkpoint, an optional baseline checkpoint, and center-frame analysis.

## Samples File

The analysis samples should be defined outside the training options:

```json
{
  "samples": [
    {
      "clip": "GOPR0384_11_02",
      "frame": "001301",
      "frame_index": 3,
      "mask": {
        "type": "box",
        "xyxy": [120, 80, 220, 160],
        "label": "motion_boundary"
      },
      "reason": "fast motion edge"
    }
  ]
}
```

The first version should support manual box masks. Later versions may add automatic masks based on motion magnitude, frame difference, or baseline error top-k regions.

Recommended sample coverage:

- Fast motion boundaries.
- Fine texture or text regions.
- Large displacement or occlusion.
- Low-light or weak-texture regions.
- At least one failure case.

## Architecture

The toolkit should live under `scripts/analysis/` and be composed of small modules:

```text
scripts/analysis/fusion_attribution.py
scripts/analysis/fusion_attr/
  probes.py
  targets.py
  cam.py
  perturb.py
  panels.py
  io.py
```

The entry point should:

1. Read and snapshot the S-VRT option file.
2. Build the model through the existing S-VRT model/network selection path.
3. Load the requested checkpoint.
4. Build the validation/test dataset from the option file.
5. Select samples from `fusion_samples.json`.
6. Attach fusion probes.
7. Run eval forward passes and attribution backward passes.
8. Export raw maps, visual maps, panels, and metadata.

## Probe Contract

The generic probe attaches around the model's fusion adapter, typically `model.netG.fusion_adapter` or the equivalent path used by the selected network.

It should capture:

```text
rgb_input
spike_input
fusion_output
module_name
placement
operator_name
mode
tensor_shape
```

This generic capture path must work for gated, concat, Mamba, PASE, and unknown future fusion operators, as long as the model uses the existing fusion adapter/operator structure.

## Optional Operator Explanation

Fusion operators may optionally expose richer internals through:

```python
def explain(self) -> dict[str, torch.Tensor]:
    return {
        "gate": ...,
        "correction": ...,
        "spike_branch": ...,
        "rgb_branch": ...,
        "temporal_weight": ...,
    }
```

This interface is optional. The toolkit must fall back to generic maps when `explain()` is unavailable.

For `GatedFusionOperator`, the most useful maps are:

```text
gate_mean        = mean_channel(gate)
correction_norm  = norm_channel(correction)
effective_update = norm_channel(gate * correction)
```

The paper-facing gated map should prefer `effective_update`, because it represents the actual injected update more directly than gate activation alone.

Fusion-specific fallback behavior:

- `gated`: export `gate_mean`, `correction_norm`, and `effective_update` when available.
- `concat`: export `fusion_delta` and optional RGB/spike channel sensitivity.
- `mamba`: export temporal spike sensitivity rather than gate-like maps.
- `pase`: export branch or modulation maps if provided by `explain()`, otherwise use generic maps.
- unknown future fusion: export generic CAM, fusion delta, and perturbation sensitivity.

## Attribution Targets

The main paper target should be a masked restoration objective:

```text
target = - Charbonnier(output_center_frame * M, gt_center_frame * M)
```

This answers which fusion-adjacent features support better restoration in a selected region.

Optional targets:

```text
error_reduction = error_baseline(M) - error_full(M)
texture_target  = mean(abs(laplacian(output_center_frame * M)))
```

`error_reduction` requires a baseline model/output. `texture_target` is for debugging or sample selection and should not be used as the main paper claim.

## Comparable Heatmap Rules

To compare fusion variants fairly:

- Use the same `fusion_samples.json`.
- Use the same frame index, defaulting to the center frame.
- Use the same mask for all variants.
- Use the same restoration target.
- Use the same target layer strategy: fusion output first, then fusion adapter output fallback.
- Use the same CAM method and normalization.
- Save raw `.npy` maps before visualization.
- Use percentile clipping, such as 1%-99%, for paper panels.
- Pair every heatmap with an error map or error reduction map.

The paper text should not claim that heatmaps prove performance. A safe caption or discussion should say that attribution maps localize restoration-sensitive regions affected by fusion, while ablation metrics and error maps establish performance changes.

## Generic Maps

Every fusion operator should support:

```text
CAM map
fusion_delta       = norm_channel(fusion_output - rgb_reference)
spike_sensitivity  = norm(output(rgb, spike) - output(rgb, perturbed_spike))
rgb_sensitivity    = norm(output(rgb, spike) - output(perturbed_rgb, spike))
error_full
error_baseline     # when baseline is provided
error_reduction    # when baseline is provided
```

`rgb_reference` is placement-dependent. For early fusion, it is the RGB input. For middle or hybrid fusion, it is the RGB feature tensor received by the adapter.

## Output Layout

Each run should create an isolated output directory:

```text
results/fusion_attribution/
  gated_2026-04-23_1530/
    config_snapshot.json
    run_manifest.json
    samples/
      GOPR0384_11_02_001301/
        panel.png
        inputs/
          blurry_rgb.png
          spike_cue.png
          gt.png
        outputs/
          restored.png
          baseline_restored.png
        maps/
          cam_raw.npy
          cam.png
          error_full.png
          error_baseline.png
          error_reduction.png
          fusion_delta.npy
          fusion_delta.png
          spike_sensitivity.npy
          spike_sensitivity.png
          gate_mean.npy
          effective_update.npy
        metadata.json
```

`metadata.json` should include:

```text
sample id
frame index
mask type and coordinates
fusion operator
fusion placement
fusion mode
target function
target layer
checkpoint path/hash
option path/hash
normalization method
CAM method
perturbation method
```

## Paper Panel

The standard paper panel should use six columns:

```text
Blurry RGB | Spike cue | Restored | Error reduction | Attribution heatmap | Fusion-specific map
```

The final column is operator-specific:

- gated: `effective_update`
- concat: `fusion_delta` or spike-channel sensitivity
- Mamba: temporal spike sensitivity
- PASE: modulation or branch map from `explain()`
- unknown future fusion: generic `fusion_delta`

A separate ablation panel may compare:

```text
Full restored | w/o fusion | baseline | Full error | w/o fusion error | error reduction
```

The attribution panel explains where fusion affects restoration. The ablation panel shows where the result improves.

## Verification Strategy

Implementation should include focused tests:

- A fake fusion adapter test confirms `rgb_input`, `spike_input`, and `fusion_output` are captured.
- A gated operator test confirms `gate_mean`, `correction_norm`, and `effective_update` are exported when available.
- An unknown operator test confirms generic fallback maps are still exported.
- A sample parsing test validates `fusion_samples.json` schema.
- A panel smoke test confirms expected output files are created.

Runtime sanity checks:

- Running the tool must not modify training options or training logs.
- Running different fusion variants with the same sample file should produce the same directory structure and comparable metadata fields.
- Raw maps should be saved before display normalization.
- A random-weight sanity check should produce attribution maps that differ from trained-weight maps.
- Perturbing high-response heatmap regions should reduce the target more than perturbing low-response regions.

## First Implementation Scope

The first version should implement:

- Generic fusion probe.
- Gated operator-specific maps.
- Manual box masks.
- Masked Charbonnier target.
- One CAM method, preferably Grad-CAM or HiResCAM depending on compatibility.
- Optional baseline model output for error reduction maps.
- Raw `.npy`, visual `.png`, `metadata.json`, and `panel.png` output.

Mamba/PASE-specific maps, automatic masks, and broader CAM metric evaluation should remain extension points.

## Paper Usage Guidance

Use heatmaps to support localization claims:

```text
The attribution maps localize restoration-sensitive regions affected by each fusion variant.
```

Use quantitative ablations and error maps to support performance claims:

```text
Combined with ablation metrics and error-reduction maps, the visualizations show that fusion primarily benefits motion boundaries and high-frequency structures.
```

Avoid saying:

```text
The heatmap proves that fusion improves performance.
```
