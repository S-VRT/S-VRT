# VRT RGB+Spike 快速入门

本文档面向当前 `baseline` 分支的 RGB+Spike 训练、验证和调试流程。旧路径 `options/vrt/...` 已不再作为主入口；当前配置集中在仓库根目录的 `options/*.json`。

## 1. 环境准备

```bash
cd /root/projects/S-VRT
uv pip list >/dev/null
```

若不用 `uv`，确保当前 Python 环境已安装 PyTorch、OpenCV、einops、pytest 以及项目依赖。

常用权重位置：

```text
weights/vrt/006_VRT_videodeblurring_GoPro.pth
weights/optical_flow/dt10_e40.pth
model_zoo/vrt/spynet_sintel_final-3d2a1287.pth
```

## 2. 选择配置

### 轻量 concat baseline

`options/006_train_vrt_videodeblurring_gopro_rgbspike.json`

- Dataset: `TrainDatasetRGBSpike`
- 输入打包: `input_pack_mode="concat"`
- Spike 表征: TFP, `spike_channels=4`
- 模型入口: `netG.in_chans=7`
- 光流: SpyNet
- 数据路径: 相对路径 `trainsets/...`

### 服务器 fusion 训练

`options/gopro_rgbspike_server.json`

- 输入打包: `input_pack_mode="dual"`
- 兼容输出: `compat.keep_legacy_L=true`
- Fusion: `netG.input.strategy="fusion"`, `netG.input.mode="dual"`
- 光流: SCFlow, 需要 `encoding25` artifact
- 预计算 Spike: `spike.precomputed.enable=true`
- 数据路径: `/root/autodl-tmp/datasets/gopro_spike/...`

### PASE residual 变体

`options/gopro_rgbspike_server_pase_residual.json`

- Fusion operator: `pase_residual`
- Frame contract: `collapsed`
- Raw ingress: `3 + 4 = 7`
- 适合从 GoPro VRT 权重做 warmup/LoRA 训练

### Raw-window Mamba 变体

`options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json`

- Spike 表征: `raw_window`
- `raw_window_length=21`
- Raw ingress: `3 + 21 = 24`
- Fusion operator: `dual_scale_temporal_mamba`
- SCFlow 子帧数: `spike_flow.subframes=21`

## 3. 数据目录检查

相对路径 baseline：

```bash
ls trainsets/GoPro/train_GT
ls trainsets/GoPro/train_GT_blurred
ls trainsets/gopro_spike/GOPRO_Large_spike_seq/train
```

服务器路径：

```bash
ls /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large/train_GT
ls /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large/train_GT_blurred
ls /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train
```

每个 Spike clip 应包含：

```text
<dataroot_spike>/<clip>/spike/000001.dat
```

如果启用预计算 TFP、raw-window 或 SCFlow，还应有对应 artifact：

```text
<clip>/tfp_b4_hw20/000001.npy
<clip>/raw_window_l21/000001.npy
<clip>/encoding25_dt10_s4/000001.npy
```

## 4. 预计算 Spike Artifact

TFP 预计算：

```bash
uv run python scripts/data_preparation/spike/prepare_spike_tfp.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --spike-h 360 \
  --spike-w 640 \
  --num-bins 4 \
  --half-win-length 20 \
  --workers 32
```

Raw-window 预计算：

```bash
uv run python scripts/data_preparation/spike/prepare_spike_raw_window.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --spike-h 360 \
  --spike-w 640 \
  --window-length 21 \
  --workers 32
```

SCFlow encoding25 子帧：

```bash
uv run python scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --dt 10 \
  --subframes 4 \
  --spike-h 360 \
  --spike-w 640 \
  --num-workers 16
```

`subframes` 必须与对应配置的 `spike_channels` 一致。Raw-window 21 通道配置需要 `--subframes 21`。

## 5. 快速验证 Dataset

Concat baseline：

```bash
uv run python - <<'PY'
from utils import utils_option as option
from data.select_dataset import define_Dataset

opt = option.parse("options/006_train_vrt_videodeblurring_gopro_rgbspike.json", is_train=True)
ds = define_Dataset(opt["datasets"]["train"])
sample = ds[0]
print(sample["L"].shape)  # [T, 7, H, W]
print(sample["H"].shape)  # [T, 3, H, W]
PY
```

Dual+fusion 服务器配置：

```bash
uv run python - <<'PY'
from utils import utils_option as option
from data.select_dataset import define_Dataset

opt = option.parse("options/gopro_rgbspike_server_pase_residual.json", is_train=True)
ds = define_Dataset(opt["datasets"]["train"])
sample = ds[0]
print(sample["L_rgb"].shape)        # [T, 3, H, W]
print(sample["L_spike"].shape)      # [T, 4, H, W]
print(sample["L"].shape)            # [T, 7, H, W] when keep_legacy_L=true
print(sample["L_flow_spike"].shape) # [T*S, 25, H, W] when SCFlow is enabled
PY
```

## 6. 启动训练

推荐使用 `launch_train.sh`，它会选择可用 GPU、设置 Python/CUDA 环境、包装 screen/script 日志，并可选启动 AutoDL TensorBoard。

单 GPU debug：

```bash
bash launch_train.sh 1 options/gopro_rgbspike_server_debug.json --foreground
```

4 GPU 服务器训练：

```bash
bash launch_train.sh 4 options/gopro_rgbspike_server.json --detach
```

指定 GPU：

```bash
bash launch_train.sh 2 options/gopro_rgbspike_server_pase_residual.json --gpus=0,1 --detach
```

平台已注入 DDP 环境时，不要再套 `torchrun`：

```bash
uv run python -u main_train_vrt.py --opt options/gopro_rgbspike_server.json
```

本地手动 DDP：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_train_vrt.py \
  --opt options/gopro_rgbspike_server.json
```

## 7. Forward Debug

用于跑完整 inference 后执行 fusion attribution/debugger：

```bash
bash scripts/run_forward_debug_experiments.sh \
  --config options/gopro_rgbspike_server_debug.json \
  --checkpoint weights/vrt/006_VRT_videodeblurring_GoPro.pth \
  --run-id debug_smoke \
  --max-samples 1
```

输出位置：

```text
<exp_dir>/images/forward_debug/<RUN_ID>/<CHECKPOINT_STEM>/inference_results
<exp_dir>/images/forward_debug/<RUN_ID>/<CHECKPOINT_STEM>/debugger
results/<task>_forward_debug_<CHECKPOINT_STEM>_<RUN_ID> -> inference_results
```

脚本会拒绝覆盖已有的 `results/...` 路径。换 `RUN_ID` 或手动移动旧路径。

## 8. 常见配置关系

| 场景 | Dataset 输出 | `netG.input` | Backbone 宽度 |
|------|--------------|--------------|---------------|
| concat TFP4 | `L=[T,7,H,W]` | `strategy` 默认 concat | `backbone_in_chans=7` |
| fusion TFP4 | `L_rgb`, `L_spike`, `L` | `strategy=fusion`, `mode=dual`, `raw_ingress_chans=7` | early fusion 后通常为 3 |
| raw-window21 | `L_spike=[T,21,H,W]` | `raw_ingress_chans=24` | early fusion 后通常为 3 |

`netG.in_chans` 和 `netG.input.raw_ingress_chans` 表示原始入口宽度，也就是 `3 + spike_channels`。启用 early fusion 后，VRT 主干实际看到的宽度由 `fusion.out_chans` 决定，代码中记录为 `backbone_in_chans`。

## 9. 常见问题

### `FileNotFoundError` 或缺少 Spike artifact

检查 `dataroot_spike`、`spike_h/spike_w` 和 artifact 目录名。启用 `spike.precomputed.enable=true` 时，TFP 目录为 `tfp_b<num_bins>_hw<half_win_length>`。

### SCFlow temporal dim mismatch

`spike_flow.subframes` 必须等于 `spike_channels`。例如 TFP4 使用 `subframes=4`，raw-window21 使用 `subframes=21`。

### Channel mismatch after SGP alignment

检查以下字段是否一致：

- `datasets.*.spike_channels`
- `datasets.*.spike.representation`
- `datasets.*.spike.raw_window_length`
- `netG.in_chans`
- `netG.input.raw_ingress_chans`
- `fusion.out_chans`

### CUDA OOM

优先降低：

- `datasets.train.dataloader_batch_size`
- `datasets.train.gt_size`
- `val.size_patch_testing`
- `num_frame_testing`

## 10. 快速检查清单

- [ ] 当前分支包含最新 `baseline` 文档和配置。
- [ ] RGB/GT/Spike 路径存在。
- [ ] 需要的 VRT/SCFlow/SpyNet 权重存在。
- [ ] 若启用预计算，`tfp_b*`、`raw_window_l*`、`encoding25_dt*_s*` 已生成。
- [ ] Dataset smoke test 能打印预期 shape。
- [ ] `spike_channels == spike_flow.subframes`。
- [ ] `netG.in_chans == netG.input.raw_ingress_chans == 3 + spike_channels`。
