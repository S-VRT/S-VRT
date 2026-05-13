# RGB+Spike 实现说明

本文档记录当前 RGB+Spike 实现的文件职责、配置入口、预处理脚本和验证命令。它是实现维护文档，不再描述旧的 `options/vrt/...` 路径。

## 1. 当前能力

当前代码支持：

- RGB-only VRT baseline。
- RGB+Spike concat 输入。
- RGB+Spike dual 输入，保留 legacy `L` 兼容。
- TFP Spike 表征。
- Raw-window Spike 表征。
- SpyNet RGB 光流。
- SCFlow Spike `encoding25` 光流。
- Early fusion operators: `gated`、`attention`、`pase_residual`、`dual_scale_temporal_mamba`。
- Expanded 和 collapsed 两类 early-fusion 时间轴契约。
- SCFlow collapsed 子帧策略 `mean_windows` 和 `compose_subframes`。
- Forward debug/inference artifact 目录和 `results/` symlink。

## 2. 文件结构

### Dataset

| 文件 | 职责 |
|------|------|
| `data/select_dataset.py` | 根据 `dataset_type` 和 `phase` 选择 dataset 类 |
| `data/dataset_video_train_rgbspike.py` | 训练 RGB+Spike dataset |
| `data/dataset_video_test.py` | 测试/验证 RGB+Spike dataset |
| `data/spike_recc/__init__.py` | Spike 工具导出 |
| `data/spike_recc/raw_window.py` | centered raw-window 提取 |
| `data/spike_recc/encoding25.py` | SCFlow encoding25 artifact 路径、shape 校验、子中心工具 |

### 模型

| 文件 | 职责 |
|------|------|
| `models/architectures/vrt/vrt.py` | VRT 输入策略、fusion、光流、SGP 对齐、输出 reducer |
| `models/fusion/factory.py` | fusion operator/adapter 创建 |
| `models/fusion/adapters.py` | early/middle/hybrid fusion adapter |
| `models/fusion/operators/` | gated、attention、PASE residual、dual-scale mamba operator |
| `models/fusion/reducers.py` | expanded 时间轴 restoration 输出 reducer |
| `models/optical_flow/` | SpyNet/SCFlow/SEA-RAFT 工厂和 wrapper |
| `models/model_plain.py` | 训练/测试时 `L`、`L_rgb`、`L_spike`、`L_flow_spike` 喂入模型 |

### 预处理和调试

| 文件 | 职责 |
|------|------|
| `scripts/data_preparation/spike/prepare_spike_tfp.py` | 生成 `tfp_b<num_bins>_hw<half_win_length>` |
| `scripts/data_preparation/spike/prepare_spike_raw_window.py` | 生成 `raw_window_l<length>` |
| `scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py` | 生成 `encoding25_dt<dt>[_s<subframes>]` |
| `scripts/run_forward_debug_experiments.sh` | 跑 inference 后执行 fusion attribution/debugger |
| `scripts/analysis/fusion_attribution.py` | fusion attribution CLI |

## 3. Dataset 实现

### 3.1 初始化配置

`TrainDatasetRGBSpike` 解析以下关键字段：

```json
{
  "dataset_type": "TrainDatasetRGBSpike",
  "dataroot_gt": "...",
  "dataroot_lq": "...",
  "dataroot_spike": "...",
  "spike_h": 360,
  "spike_w": 640,
  "spike_channels": 4,
  "input_pack_mode": "dual",
  "compat": {
    "keep_legacy_L": true
  },
  "spike": {
    "representation": "tfp",
    "reconstruction": {
      "type": "spikecv_tfp",
      "num_bins": 4
    },
    "precomputed": {
      "enable": true,
      "root": "auto",
      "format": "npy"
    }
  },
  "spike_flow": {
    "representation": "encoding25",
    "dt": 10,
    "root": "auto",
    "subframes": 4
  }
}
```

兼容字段仍存在，例如 `spike_reconstruction`、`middle_tfp_center`、`keep_legacy_l`。新配置优先使用 `spike.*` 和 `compat.*`。

### 3.2 `__getitem__` 数据流

1. 根据 `meta_info_file` 选取相邻帧列表。
2. 读取 RGB LQ/GT。
3. 读取或预计算加载 Spike restoration 表征。
4. 若启用 SCFlow，读取 `encoding25` flow Spike。
5. 对 RGB 做 paired random crop。
6. 将 crop 参数映射到 Spike/SCFlow 源分辨率。
7. crop+resize Spike 到 RGB crop 尺寸。
8. 合并 RGB+Spike 为 legacy `L`。
9. 对 `L`、`L_flow_spike`、GT 做一致 augment。
10. 根据 `input_pack_mode` 返回 concat 或 dual keys。

返回契约：

```text
concat:
  L: [T,3+S,H,W]
  H: [T,3,H,W]

dual:
  L_rgb: [T,3,H,W]
  L_spike: [T,S,H,W]
  L: [T,3+S,H,W] when keep_legacy_L=true
  H: [T,3,H,W]
  L_flow_spike: [T*S,25,H,W] when SCFlow is enabled
```

## 4. Spike 表征实现

### 4.1 TFP

在线路径调用：

```python
voxelize_spikes_tfp(spike_matrix, num_channels=spike_channels, half_win_length=tfp_half_win_length)
```

预计算路径查找：

```text
<dataroot_spike>/<clip>/tfp_b<num_bins>_hw<half_win_length>/<frame>.npy
```

生成命令：

```bash
uv run python scripts/data_preparation/spike/prepare_spike_tfp.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --spike-h 360 \
  --spike-w 640 \
  --num-bins 4 \
  --half-win-length 20 \
  --workers 32
```

### 4.2 Raw Window

在线路径调用：

```python
extract_centered_raw_window(spike_matrix, window_length=raw_window_length)
```

预计算路径查找：

```text
<dataroot_spike>/<clip>/raw_window_l<raw_window_length>/<frame>.npy
```

生成命令：

```bash
uv run python scripts/data_preparation/spike/prepare_spike_raw_window.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --spike-h 360 \
  --spike-w 640 \
  --window-length 21 \
  --workers 32
```

## 5. SCFlow 实现

Dataset 加载 `spike_flow.representation="encoding25"` 时：

```text
subframes = 1:
  <clip>/encoding25_dt10/<frame>.npy -> [25,H,W]

subframes > 1:
  <clip>/encoding25_dt10_s4/<frame>.npy -> [4,25,H,W]
```

`__getitem__` 会将每个 RGB frame 的子窗口展平：

```text
T frames * S subframes -> [T*S,25,H,W]
```

生成命令：

```bash
uv run python scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --dt 10 \
  --subframes 4 \
  --spike-h 360 \
  --spike-w 640 \
  --num-workers 16
```

`spike_flow.collapse_policy` 在 VRT 中解析：

- `mean_windows`: 默认，将 `[B,T,S,25,H,W]` 均值到 `[B,T,25,H,W]`。
- `compose_subframes`: 在 `[B,T*S,25,H,W]` 上跑 SCFlow，再组合到 RGB 帧级 flow。

train/test split 的 `collapse_policy` 必须一致。

## 6. VRT 输入实现

`models/architectures/vrt/vrt.py` 解析：

```json
"netG": {
  "input": {
    "strategy": "fusion",
    "mode": "dual",
    "raw_ingress_chans": 7
  },
  "fusion": {
    "enable": true,
    "placement": "early",
    "operator": "pase_residual",
    "mode": "replace",
    "out_chans": 3,
    "early": {
      "frame_contract": "collapsed"
    }
  },
  "in_chans": 7
}
```

关键属性：

```text
self.in_chans             raw ingress width, 3 + spike_channels
self.backbone_in_chans    actual VRT backbone width after early/hybrid fusion
self.spike_bins           spike_channels when fusion is enabled
self._last_fusion_meta    diagnostics from fusion adapter/operator
```

`conv_first` 通道：

```text
pa_frames > 0 and nonblind=false:
  conv_first.in_channels = backbone_in_chans * 9
```

Concat TFP4:

```text
backbone_in_chans = 7
conv_first.in_channels = 63
```

Early fusion TFP4 with `out_chans=3`:

```text
raw_ingress_chans = 7
backbone_in_chans = 3
conv_first.in_channels = 27
```

## 7. 配置入口

| 配置 | 用途 |
|------|------|
| `options/006_train_vrt_videodeblurring_gopro_rgbspike.json` | TFP4 concat + SpyNet baseline |
| `options/gopro_rgbspike_server.json` | TFP4 dual+fusion + SCFlow server 主配置 |
| `options/gopro_rgbspike_server_debug.json` | server debug 配置 |
| `options/gopro_rgbspike_server_pase_residual.json` | PASE residual operator |
| `options/gopro_rgbspike_server_pase_residual_snapshot.json` | two-run PASE residual snapshot |
| `options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json` | raw-window21 + dual-scale temporal mamba |
| `options/gopro_server_mamba_collapsed_scflow_snapshot.json` | collapsed SCFlow mamba snapshot |
| `options/gopro_server_mamba_expanded_scflow_snapshot.json` | expanded SCFlow mamba snapshot |
| `options/gopro_server_pase_scflow.json` | PASE + SCFlow config |
| `options/gopro_server_pase_scflow_snapshot.json` | PASE + SCFlow snapshot |
| `options/gopro_server_pase_spynet_snapshot.json` | PASE + SpyNet snapshot |

## 8. 验证命令

Dataset 和 raw-window：

```bash
uv run pytest tests/data/test_dataset_rgbspike_raw_window.py \
              tests/data/test_prepare_spike_raw_window.py \
              tests/data/test_spike_raw_window.py -v
```

SCFlow contract：

```bash
uv run pytest tests/models/test_optical_flow_scflow_contract.py \
              tests/models/test_optical_flow_scflow_integration.py -v
```

Fusion/VRT integration：

```bash
uv run pytest tests/models/test_fusion_early_adapter.py \
              tests/models/test_vrt_fusion_integration.py -v
```

Forward debug script tests：

```bash
uv run pytest tests/test_forward_debug_batch_script.py -v
```

Server config smoke：

```bash
uv run pytest tests/e2e/test_gopro_rgbspike_server_e2e.py -v
```

## 9. 常见错误和定位

### `raw_window_length` 与 `spike_channels` 冲突

Raw-window 模式下：

```text
spike_channels == spike.raw_window_length
```

例如 `raw_window_length=21` 时，`netG.in_chans` 和 `netG.input.raw_ingress_chans` 应为 `24`。

### SCFlow 缺少 artifact

错误通常提示运行：

```text
scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py
```

确认目录名是否匹配 `dt` 和 `subframes`：

```text
encoding25_dt10
encoding25_dt10_s4
encoding25_dt10_s21
```

### SCFlow 时间轴不匹配

检查：

```text
datasets.train.spike_channels
datasets.train.spike_flow.subframes
datasets.test.spike_channels
datasets.test.spike_flow.subframes
fusion.early.frame_contract
spike_flow.collapse_policy
```

### SGP channel mismatch

错误来自 `get_aligned_image_2frames` 或 `conv_first` 前。检查：

```text
raw_ingress_chans = 3 + spike_channels
backbone_in_chans = raw_ingress_chans or fusion.out_chans
conv_first.in_channels = backbone_in_chans * 9
```

### SpyNet 权重与 SCFlow 权重混用

`netG.optical_flow.module="spynet"` 使用 RGB flow checkpoint；`"scflow"` 使用 Spike flow checkpoint。二者输入类型不同，不能只替换 checkpoint。

## 10. 维护原则

- 新实验优先使用 `dual+fusion`，除非明确要复现 concat baseline。
- 新 Spike restoration 表征必须在 Dataset 中明确 shape，并在 VRT/fusion 中明确 raw ingress 宽度。
- SCFlow 输入始终走 `L_flow_spike`，不要把 restoration Spike 通道直接喂给 SCFlow。
- 配置中 train/test 的 Spike 表征、`spike_flow.subframes` 和 `collapse_policy` 应保持一致。
- 修改输入契约时同步更新 `docs/QUICKSTART_RGB_SPIKE.md`、`docs/RGB_SPIKE_DESIGN.md` 和 `docs/SCFLOW_SUBFRAME_FUSION.md`。
