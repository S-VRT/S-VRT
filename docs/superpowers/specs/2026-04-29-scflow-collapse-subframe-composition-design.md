# SCFlow Collapse Subframe Composition Design

## Context

S-VRT currently supports early-fusion frame contracts:

- `expanded`: the backbone runs on `N*S` temporal steps, where `S` is the number of Spike bins/subframes.
- `collapsed`: the backbone runs on the original RGB frame count `N`.

SCFlow uses dedicated `encoding25` Spike flow input, separate from the restoration Spike channels. With `spike_flow.subframes > 1`, the dataset can provide `L_flow_spike` as `[B, N*S, 25, H, W]`.

The current collapsed path reduces `[B, N*S, 25, H, W]` to `[B, N, 25, H, W]` by averaging subframe Spike windows before SCFlow. That preserves the model input shape but discards the fine temporal motion cues that subframes were meant to provide.

## Existing Spatial Alignment

The project currently keeps RGB and Spike spatial processing consistent under the assumption that both streams share the same field of view and differ only by resolution:

- Training uses the RGB crop parameters, maps them by `src_h / lq_h_orig` and `src_w / lq_w_orig` into Spike coordinates, crops the corresponding Spike region, then resizes Spike tensors to the RGB crop size.
- Validation and lazy patch loading use the same proportional mapping from RGB full-frame/patch coordinates into Spike source coordinates, then resize to the RGB target size.
- VRT rejects SCFlow inputs whose spatial size does not match the backbone tensor.

This does not perform camera calibration, homography correction, or distortion correction. That is acceptable for the current data assumption: original RGB and Spike are same-view streams with linear scale differences.

## Goal

When early fusion is collapsed, keep the backbone at RGB frame resolution `N`, but make the alignment flow benefit from fine-grained Spike subframes.

The collapsed path should estimate flow on the `N*S` Spike flow timeline, then compose those subframe flows into RGB frame-level flows used by the existing VRT alignment path.

## Non-Goals

- Do not expand the collapsed backbone to `N*S`.
- Do not introduce calibration, homography, or distortion models.
- Do not change the dataset artifact format.
- Do not change SCFlow itself.

## Proposed Behavior

Add a configurable collapse policy for SCFlow subframe inputs:

```json
"spike_flow": {
  "collapse_policy": "compose_subframes"
}
```

Supported policies:

- `mean_windows`: current compatibility behavior. Average `[B, N, S, 25, H, W]` over `S` before SCFlow.
- `compose_subframes`: new behavior. Estimate SCFlow on the expanded `N*S` timeline, then compose subframe flows back to `[B, N-1, 2, H, W]`.

The default should remain `mean_windows` initially to avoid changing existing experiments silently. New experiments that need fine Spike motion in collapsed mode should set `compose_subframes`.

## Flow Composition

For collapsed backbone input `x: [B, N, C, H, W]` and subframe flow input `flow_spike: [B, N*S, 25, H, W]`:

1. Run SCFlow on adjacent subframe Spike windows to obtain:
   - forward subframe flows: `[B, N*S-1, 2, H, W]`
   - backward subframe flows: `[B, N*S-1, 2, H, W]`
2. Choose each RGB frame's representative subframe index as the center bin:
   - `anchor = S // 2`
   - frame `t` maps to subframe `t*S + anchor`
3. Compose forward flow from `t*S + anchor` to `(t+1)*S + anchor`.
4. Compose backward flow from `(t+1)*S + anchor` to `t*S + anchor`.
5. Return composed frame-level flows with the existing VRT multiscale list contract.

Composition uses the existing VRT convention:

```python
F_a_to_c = F_a_to_b + flow_warp(F_b_to_c, F_a_to_b.permute(0, 2, 3, 1))
```

This is physically more meaningful than averaging windows or averaging flows because each incremental displacement is evaluated in the coordinate system reached by previous displacements.

## Architecture

Implement the behavior inside VRT's flow path rather than in the dataset:

- Dataset remains responsible for loading and spatially aligning `L_flow_spike`.
- VRT detects the mismatch `flow_spike.size(1) == x.size(1) * spike_bins` in collapsed mode.
- VRT chooses the collapse policy from config.
- `get_flow_2frames()` dispatches to a new helper for `compose_subframes`.
- Existing `get_aligned_image_2frames()` continues to consume frame-level `[B, N-1, 2, H, W]` flows unchanged.

This keeps the change local to flow semantics and avoids coupling fusion operators to optical-flow details.

## Error Handling

- `compose_subframes` requires `spike_bins > 1`.
- `compose_subframes` requires `flow_spike.size(1) == x.size(1) * spike_bins`.
- If the shape does not match, raise a clear `ValueError` rather than falling back silently.
- Preserve current SCFlow validation for channel count `25` and matching spatial size.

## Tests

Add focused tests for:

- Collapsed + `mean_windows` preserves existing behavior.
- Collapsed + `compose_subframes` passes expanded `N*S` Spike inputs to SCFlow rather than averaged `[N]` windows.
- Composed forward/backward frame-level outputs have shape `[B, N-1, 2, H, W]`.
- A deterministic fake flow backend verifies composition order and anchor selection.
- Spatial mismatch still raises before SCFlow.

## Open Decisions

The anchor will be `S // 2` for the first implementation. This matches the current expanded path's center-subframe reduction convention. Later experiments can add an anchor policy if needed.
