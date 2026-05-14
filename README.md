# S-VRT: Spike-Enhanced Video Restoration Transformer

[English](README.md) | [简体中文](README_zh-CN.md)

S-VRT is a research codebase for video deblurring and restoration with RGB frames and Spike camera streams. It extends the VRT/KAIR training stack with RGB+Spike datasets, Spike restoration representations, structured fusion operators, and optional Spike-based optical flow through SCFlow.

The default README is intentionally concise. Detailed RGB+Spike and SCFlow documentation lives under [`docs/`](docs/).

## Highlights

- **RGB+Spike restoration**: train VRT-style restoration models with RGB frames plus Spike camera data.
- **Two input contracts**: legacy `concat` input and structured `dual+fusion` input.
- **Spike representations**: TFP voxel bins and centered raw Spike windows.
- **Fusion operators**: gated, attention, PASE residual, and dual-scale temporal Mamba variants.
- **Optical flow choices**: RGB SpyNet or Spike SCFlow with `encoding25` artifacts.
- **SCFlow subframes**: `mean_windows` and `compose_subframes` policies for collapsed early fusion.
- **Training utilities**: single-node DDP launch script, dataset preparation helpers, TensorBoard/Weights & Biases/SwanLab/Logfire hooks, and forward-debug scripts.

## Repository Status

Current RGB+Spike configs are in [`options/`](options/), not `options/vrt/`.

Important entry configs:

| Config | Purpose |
|--------|---------|
| [`options/006_train_vrt_videodeblurring_gopro_rgbspike.json`](options/006_train_vrt_videodeblurring_gopro_rgbspike.json) | TFP4 concat + SpyNet baseline |
| [`options/gopro_rgbspike_server.json`](options/gopro_rgbspike_server.json) | Server RGB+Spike dual+fusion + SCFlow config |
| [`options/gopro_rgbspike_server_debug.json`](options/gopro_rgbspike_server_debug.json) | Debug-sized server config |
| [`options/gopro_rgbspike_server_pase_residual.json`](options/gopro_rgbspike_server_pase_residual.json) | PASE residual fusion variant |
| [`options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json`](options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json) | Raw-window21 + dual-scale temporal Mamba variant |

## Installation

The project is normally run inside a prepared Python/PyTorch environment. From the repository root:

```bash
cd /root/projects/S-VRT
uv pip list >/dev/null
```

If you do not use `uv`, install the project dependencies in your Python environment. Core runtime dependencies include PyTorch, OpenCV, NumPy, einops, timm, tensorboard, wandb, swanlab, and pytest.

Optional CUDA extensions such as DCNv4 must be built only when the selected config requires them.

## Model Weights

Download the project checkpoint from Hugging Face:

- Spk-VRT LoRA checkpoint: <https://huggingface.co/vlmbgnr/Spk-VRT_LoRA_10000_iter/tree/main>

Place the downloaded Spk-VRT checkpoint files under:

```text
weights/SpkVRT/
```

The VRT GoPro video deblurring pretrained weight is hosted in the upstream VRT releases:

- VRT releases: <https://github.com/JingyunLiang/VRT/releases>

Download `006_VRT_videodeblurring_GoPro.pth` and place it under:

```text
weights/vrt/
```

## Data Layout

The expected GoPro + Spike layout is:

```text
gopro_spike/
├── GOPRO_Large/
│   ├── train_GT/
│   ├── train_GT_blurred/
│   ├── test_GT/
│   └── test_GT_blurred/
└── GOPRO_Large_spike_seq/
    ├── train/
    │   └── <clip>/
    │       └── spike/
    │           └── *.dat
    ├── test/
    └── config.yaml
```

Server configs currently point to:

```text
/root/autodl-tmp/datasets/gopro_spike/GOPRO_Large
/root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq
```

## Spike Artifact Preparation

TFP artifacts:

```bash
uv run python scripts/data_preparation/spike/prepare_spike_tfp.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --spike-h 360 \
  --spike-w 640 \
  --num-bins 4 \
  --half-win-length 20 \
  --workers 32
```

Raw-window artifacts:

```bash
uv run python scripts/data_preparation/spike/prepare_spike_raw_window.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --spike-h 360 \
  --spike-w 640 \
  --window-length 21 \
  --workers 32
```

SCFlow `encoding25` artifacts:

```bash
uv run python scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --dt 10 \
  --subframes 4 \
  --spike-h 360 \
  --spike-w 640 \
  --num-workers 16
```

For raw-window21 configs, generate SCFlow artifacts with `--subframes 21`.

## Quick Start

Dataset smoke test for the concat baseline:

```bash
uv run python - <<'PY'
from utils import utils_option as option
from data.select_dataset import define_Dataset

opt = option.parse("options/006_train_vrt_videodeblurring_gopro_rgbspike.json", is_train=True)
ds = define_Dataset(opt["datasets"]["train"])
sample = ds[0]
print(sample["L"].shape)
print(sample["H"].shape)
PY
```

Run a debug training job:

```bash
bash launch_train.sh 1 options/gopro_rgbspike_server_debug.json --foreground
```

Run a detached multi-GPU training job:

```bash
bash launch_train.sh 4 options/gopro_rgbspike_server.json --detach
```

If your platform already injects DDP environment variables, do not wrap the command in `torchrun`:

```bash
uv run python -u main_train_vrt.py --opt options/gopro_rgbspike_server.json
```

## Testing

Focused documentation and contract checks:

```bash
uv run pytest tests/test_forward_debug_batch_script.py -v
uv run pytest tests/models/test_optical_flow_scflow_contract.py tests/models/test_optical_flow_scflow_integration.py -v
uv run pytest tests/models/test_fusion_early_adapter.py tests/models/test_vrt_fusion_integration.py -v
```

Server config smoke tests:

```bash
uv run pytest tests/e2e/test_gopro_rgbspike_server_e2e.py -v
```

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/QUICKSTART_RGB_SPIKE.md`](docs/QUICKSTART_RGB_SPIKE.md) | Current RGB+Spike operational guide |
| [`docs/RGB_SPIKE_DESIGN.md`](docs/RGB_SPIKE_DESIGN.md) | Data/model contracts and design rationale |
| [`docs/RGB_SPIKE_IMPLEMENTATION.md`](docs/RGB_SPIKE_IMPLEMENTATION.md) | Implementation map, scripts, configs, validation |
| [`docs/SCFLOW_SUBFRAME_FUSION.md`](docs/SCFLOW_SUBFRAME_FUSION.md) | SCFlow subframe integration and collapse policies |
| [`docs/SGP_MODULE.md`](docs/SGP_MODULE.md) | SGP module notes |
| [`docs/SPIKE_ENCODER_DESIGN.md`](docs/SPIKE_ENCODER_DESIGN.md) | Spike encoder design notes |

## Project Layout

```text
S-VRT/
├── main_train_vrt.py
├── main_test_vrt.py
├── launch_train.sh
├── launch_test.sh
├── data/
│   ├── dataset_video_train_rgbspike.py
│   ├── dataset_video_test.py
│   └── spike_recc/
├── models/
│   ├── architectures/vrt/
│   ├── fusion/
│   └── optical_flow/
├── options/
├── scripts/
│   ├── data_preparation/
│   └── analysis/
├── tests/
└── docs/
```
