# spkvisual Flow Observation Design

## Goal

Build a server-runnable observation tool under `spkvisual/` that compares SpyNet RGB flow and SCFlow spike flow on real GoPro RGB/spike samples, then persists every console conclusion as artifacts.

## Scope

The tool reads an existing option file, dataset split, and frame metadata. It samples adjacent frame pairs, runs SpyNet on RGB LQ frames and SCFlow on corresponding `encoding25` spike windows, and compares full-resolution 2-channel flow outputs after resizing to a common spatial size.

It does not change training, VRT alignment, DCN offsets, or model wrappers.

## Inputs

- Option JSON path, defaulting to `options/gopro_rgbspike_server.json`.
- Dataset split, defaulting to `train`.
- Sample count and start index.
- SpyNet checkpoint, defaulting to `weights/optical_flow/spynet/spynet_sintel_final-3d2a1287.pth`.
- SCFlow checkpoint, defaulting to `weights/optical_flow/dt10_e40.pth`.
- Device, defaulting to `cuda` when available.
- Subframe selection for `encoding25` tensors when `spike_flow.subframes > 1`: default `middle`, with `mean` and integer index also supported.

## Outputs

Each run writes to `spkvisual/flow_observations/<timestamp>/` unless `--out` is provided.

- `summary.json`: persisted version of the console conclusions, including per-model magnitude statistics, low-flow ratios, active/inactive spike-region statistics, and SCFlow-vs-SpyNet difference statistics.
- `per_pair.csv`: one row per adjacent pair with key, paths, model statistics, and pairwise difference statistics.
- `hist_flow_mag.png`: SCFlow and SpyNet magnitude histogram.
- `hist_u_v.png`: per-component flow histograms.
- `diff_mag_hist.png`: histogram of absolute flow-vector difference magnitude.

## Metrics

For each flow source:

- Mean, standard deviation, min, max.
- Percentiles: p01, p05, p25, p50, p75, p95, p99.
- Low-flow ratio under a configurable magnitude threshold.
- Active spike region and inactive spike region flow-magnitude summaries, where activity is derived from the selected SCFlow input window.

For SCFlow vs SpyNet:

- Difference magnitude summary after resizing SpyNet to SCFlow resolution.
- Signed `u` and `v` difference summaries.
- Cosine similarity summary where both vectors have non-trivial magnitude.

## Error Handling

The script fails fast with actionable messages for missing RGB frames, missing `encoding25` artifacts, unsupported artifact formats, and invalid subframe selection. It records run configuration in `summary.json` so copied console logs are not the only evidence.

## Testing

Unit tests cover the statistics helpers, subframe selector, JSON comment stripping, and artifact summary structure without requiring CUDA, checkpoints, or the server dataset.
