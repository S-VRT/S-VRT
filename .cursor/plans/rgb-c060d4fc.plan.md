<!-- c060d4fc-c8b6-43fd-9a0f-c901d839d952 43a26a82-6fef-4b61-90ca-1a14cce484a8 -->
# SpikeTFP 7-Channel Enablement Plan

## 1. Lock SGP/Encoder Channel Logic (`models/network_vrt.py`)

- Reset `conv_first_in_chans` to the explicit `9 * in_chans` (+noise map for nonblind) so the encoder input depends only on `in_chans`.
- Audit SGP helpers (`get_aligned_image_2frames`, `flow_warp`, `get_flow_2frames`) to ensure they always use the 2-frame, `nearest4` path and that SpyNet only consumes the first 3 channels (add a helper like `extract_rgb`).
- Before `conv_first`, add assertions that the concatenated tensor’s channel count matches `self.conv_first.in_channels`, logging the current shape, configured `in_chans`, and training/testing mode for fast diagnosis.

## 2. Enforce Input Channel Consistency at Net Entrances (`models/model_plain.py`, `models/model_vrt.py`)

- When feeding data to `netG`/`netE`, assert that the temporal tensor’s channel dimension equals `opt['netG']['in_chans']` (7) and raise informative errors.
- Extend the same checks to `_test_video`/`_test_clip` so both training and validation paths abort early if RGB-only clips are accidentally constructed.

## 3. SpikeTFP Data Preparation (`data/dataset_video_train_rgbspike.py`, `utils/spike_loader.py`)

- Ensure TFP voxelization always returns 4 channels (default `spike_channels=4`), document the RGB (0–2) + Spike (3–6) ordering, and keep the tensor stacking path producing `[T, 7, H, W]` without channel drops during augmentation/cropping.
- If needed, expose simple normalization hooks so RGB uses ImageNet-style stats while spike channels take their own scaling (even if currently identity) and note this explicitly in comments.

## 4. Configuration Updates (`options/gopro_rgbspike_local*.json`, related presets)

- Set `network_G.in_chans` to 7 everywhere the RGB+Spike pipeline is used and align dataset options (`spike_channels=4`, voxelization params) plus any encoder configs referencing `in_chans`.
- Add inline comments in these JSON files describing the 3+4 channel split to prevent regressions.

## 5. Validation & Documentation (`docs/RGB_SPIKE_IMPLEMENTATION.md` or README snippet)

- Describe the repo-wide `in_chans`/shape sanity checklist (search spots, what to ignore) and spell out a minimal experiment: tiny dataset, `val_freq` lowered to trigger `model.test()`, expecting no "63 vs 36" style errors.
- Capture flow inputs vs. SGP inputs behavior so future contributors understand why RGB extraction happens while maintaining 63-channel conv input.

### To-dos

- [ ] Clamp conv_first+alignment to 9×in_chans baseline
- [ ] Add netG/netE 7ch assertions in train/test paths
- [ ] Guarantee dataset returns RGB3+SpikeTFP4 tensors
- [ ] Bump RGB+Spike configs to in_chans=7 & spike_channels=4
- [ ] Document sanity checklist & minimal validation run