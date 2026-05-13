# RGB+Spike 设计说明

本文档描述当前 `baseline` 分支中 RGB+Spike 数据流、输入契约、fusion 路径和光流路径的设计。历史的单一 "RGB+Spike concat" 方案仍可用，但当前主线同时支持 `concat`、`dual+fusion`、TFP、raw-window、SpyNet 和 SCFlow。

## 1. 核心目标

S-VRT 的 RGB+Spike 设计目标是：

1. 保持 VRT 输出为 RGB restoration: `[B, T, 3, H, W]`。
2. 允许 Spike 作为额外输入模态参与恢复。
3. 保留向后兼容的 concat 入口，便于复现实验。
4. 为新实验提供结构化 `dual+fusion` 入口，让 RGB、Spike 和 SCFlow 各走清晰的数据契约。
5. 在运行时尽早暴露通道数、时间轴和 artifact 不一致问题。

## 2. 数据表示

### 2.1 RGB

RGB 路径读取 LQ/GT 图像：

```text
dataroot_lq/<clip>/<frame>.png -> LQ RGB
dataroot_gt/<clip>/<frame>.png -> GT RGB
```

Dataset 内部通过 OpenCV 解码后转成 RGB float32，值域通常为 `[0,1]`。

### 2.2 Spike restoration 表征

Spike 原始文件：

```text
dataroot_spike/<clip>/spike/<frame>.dat
```

当前支持两类 restoration 表征。

#### TFP

配置：

```json
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
}
```

输出：

```text
spike_voxel: [S,Hs,Ws], S = spike_channels = num_bins
```

预计算目录：

```text
<clip>/tfp_b4_hw20/<frame>.npy
```

#### Raw Window

配置：

```json
"spike": {
  "representation": "raw_window",
  "raw_window_length": 21,
  "reconstruction": {
    "type": "spikecv_tfp"
  }
}
```

输出：

```text
raw_window: [L,Hs,Ws], L = raw_window_length = spike_channels
```

预计算目录：

```text
<clip>/raw_window_l21/<frame>.npy
```

`raw_window_length` 必须是正奇数。若显式设置 `spike_channels`，它必须等于 `raw_window_length`。

### 2.3 SCFlow encoding25

SCFlow 不直接复用 restoration Spike 通道，而是使用独立的 `encoding25` 表征：

```text
flow_spike: [S,25,Hs,Ws] per RGB frame when subframes > 1
```

Dataset 展平后返回：

```text
sample["L_flow_spike"]: [T*S,25,H,W]
```

`S` 来自 `spike_flow.subframes`，在 early fusion 子帧场景中必须等于 `spike_channels`。

## 3. Dataset 契约

实现文件：

```text
data/dataset_video_train_rgbspike.py
data/dataset_video_test.py
data/select_dataset.py
```

`dataset_type="TrainDatasetRGBSpike"` 会在训练 phase 使用训练 dataset，在 test phase 使用测试 dataset。

### 3.1 空间对齐

Dataset 以 RGB crop 为基准：

1. 读取一组 RGB LQ/GT 帧。
2. 对 RGB 做 paired random crop。
3. 将 RGB crop 参数按比例映射到 Spike 原始分辨率。
4. 对 Spike `[C,Hs,Ws]` 做对应 crop。
5. resize Spike 到 RGB crop 尺寸。
6. 对 RGB、Spike、flow Spike 和 GT 共同做 flip/rotate augment。

该设计假设 RGB 与 Spike 是同视场线性缩放关系；当前没有相机标定、homography 或畸变校正。

### 3.2 Concat 模式

配置：

```json
"input_pack_mode": "concat"
```

输出：

```text
sample["L"]: [T,3+S,H,W]
sample["H"]: [T,3,H,W]
```

通道顺序：

```text
0..2      RGB
3..3+S-1  Spike
```

这是 `options/006_train_vrt_videodeblurring_gopro_rgbspike.json` 的主要路径。

### 3.3 Dual 模式

配置：

```json
"input_pack_mode": "dual",
"compat": {
  "keep_legacy_L": true
}
```

输出：

```text
sample["L_rgb"]: [T,3,H,W]
sample["L_spike"]: [T,S,H,W]
sample["L"]: [T,3+S,H,W]       # keep_legacy_L=true 时保留
sample["H"]: [T,3,H,W]
sample["L_flow_spike"]: [T*S,25,H,W]  # 启用 SCFlow 时
```

Dual 模式是 fusion 主线入口。`L` 保留是为了兼容旧训练/验证路径和分析脚本。

## 4. 模型输入策略

实现文件：

```text
models/architectures/vrt/vrt.py
models/fusion/*
```

### 4.1 Concat Strategy

配置通常为：

```json
"netG": {
  "in_chans": 7,
  "input_mode": "concat"
}
```

VRT 直接消费 `[B,T,7,H,W]`。当 `pa_frames=2` 时，SGP/parallel warping 会把主干输入拼成：

```text
current x                 backbone_in_chans
backward nearest4 warp     4 * backbone_in_chans
forward nearest4 warp      4 * backbone_in_chans
conv_first input           9 * backbone_in_chans
```

TFP4 concat 时 `backbone_in_chans=7`，因此 `conv_first.in_channels=63`。

### 4.2 Fusion Strategy

配置：

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

`raw_ingress_chans` 和 `in_chans` 表示原始入口宽度，也就是 `3 + spike_channels`。启用 early/hybrid fusion 后，VRT 主干实际看到的宽度不一定等于原始入口宽度，代码中用 `backbone_in_chans` 表示。

示例：

```text
raw ingress: RGB3 + TFP4 = 7
fusion out_chans: 3
backbone_in_chans: 3
conv_first input with pa_frames=2: 27
```

### 4.3 Fusion Operators

当前主线常用 operator：

| Operator | Restoration 表征 | 默认 frame contract | 说明 |
|----------|------------------|---------------------|------|
| `gated` | TFP | expanded | 早期门控融合 |
| `pase_residual` | TFP 或 raw-window | collapsed | PASE 编码 Spike 后残差注入 RGB |
| `dual_scale_temporal_mamba` | TFP 或 raw-window | collapsed | 同时建模局部/全局 Spike 时间结构 |
| `attention` | TFP | collapsed | 注意力融合实验 |

Raw-window 目前只允许用于 `pase_residual` 和 `dual_scale_temporal_mamba`。配置不匹配会在 VRT 初始化时报错。

## 5. Frame Contract

Early fusion 有两种时间轴契约。

### Expanded

```text
RGB:   [B,T,3,H,W]
Spike: [B,T,S,H,W]
main:  [B,T*S,3,H,W]
```

适用于希望主干在子帧粒度运行的实验。输出 restoration 时会通过 reducer 回到 `[B,T,3,H,W]`。

### Collapsed

```text
RGB:   [B,T,3,H,W]
Spike: [B,T,S,H,W]
main:  [B,T,3,H,W]
```

适用于 PASE residual、dual-scale temporal mamba 等 operator。主干保持 RGB 帧粒度，Spike 的子帧/窗口信息在 fusion operator 内部聚合。

## 6. 光流路径

### 6.1 SpyNet

SpyNet 使用 RGB 输入：

```python
x_flow = self.extract_rgb(x)
```

即使 `x` 是 concat `[B,T,7,H,W]`，SpyNet 也只看前三个 RGB 通道，避免把预训练 RGB 光流网络喂入 Spike 通道。

### 6.2 SCFlow

SCFlow 使用 `sample["L_flow_spike"]`：

```text
flow_spike: [B,T,25,H,W] 或 [B,T*S,25,H,W]
```

VRT 校验：

- `flow_spike.ndim == 5`
- channel 必须为 25
- spatial size 必须等于 backbone 当前 `H,W`
- temporal size 必须能与当前 backbone 时间轴对齐

Collapsed early fusion 下，SCFlow 有两种子帧 collapse 策略：

| `spike_flow.collapse_policy` | 行为 |
|------------------------------|------|
| `mean_windows` | 默认。把 `[B,T,S,25,H,W]` 对 S 求均值后给 SCFlow |
| `compose_subframes` | 在 `T*S` 子帧时间轴上估计相邻 flow，再组合回 RGB 帧级 flow |

`compose_subframes` 的 anchor 为 `S // 2`，组合使用 `flow_warp` 逐段累加位移。

## 7. 关键配置公式

```text
spike_channels = restoration Spike 通道数
raw_ingress_chans = in_chans = 3 + spike_channels
backbone_in_chans = raw_ingress_chans         # concat
backbone_in_chans = fusion.out_chans          # early/hybrid fusion
conv_first.in_channels = backbone_in_chans * 9  # pa_frames > 0, nonblind=false
```

SCFlow 子帧：

```text
spike_flow.subframes == spike_channels
L_flow_spike temporal length = T * spike_flow.subframes
```

Raw-window：

```text
spike.representation = raw_window
spike.raw_window_length = spike_channels
raw_ingress_chans = 3 + raw_window_length
```

## 8. 运行时防护

Dataset 防护：

- raw-window 长度必须为正奇数。
- raw-window 下 `spike_channels` 必须等于 `raw_window_length`。
- TFP 下 `spike_channels` 必须等于 `spike.reconstruction.num_bins`。
- `spike_flow.subframes > 1` 时必须等于 `spike_channels`。
- `L` 通道必须等于 `3 + spike_channels`。

VRT 防护：

- `strategy=fusion` 必须配 `mode=dual`。
- fusion raw ingress 必须大于 3。
- raw-window 只允许配支持的 operator。
- `pase_residual` 和 `dual_scale_temporal_mamba` 必须使用 collapsed frame contract。
- SCFlow 输入缺失、通道错误、空间错误或时间轴错误会抛 `ValueError`。
- SGP alignment 后通道必须等于 `4 * backbone_in_chans`。

## 9. 推荐实验序列

1. `options/006_train_vrt_videodeblurring_gopro_rgbspike.json`
   - TFP4 concat + SpyNet
   - 用于验证基础 RGB+Spike dataloader 和 VRT concat 路径
2. `options/gopro_rgbspike_server_pase_residual.json`
   - TFP4 dual+fusion + SCFlow + PASE residual
   - 当前服务器主线之一
3. `options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json`
   - Raw-window21 + dual-scale temporal mamba + SCFlow
   - 用于比较 raw Spike temporal window 表征

## 10. 相关文件

| 文件 | 职责 |
|------|------|
| `data/dataset_video_train_rgbspike.py` | 训练 RGB+Spike dataset、crop/resize/augment、TFP/raw-window/SCFlow artifact 加载 |
| `data/dataset_video_test.py` | 测试 phase 的 RGB+Spike dataset 和 test dataset |
| `data/spike_recc/` | Spike `.dat` 解码、TFP、raw-window、encoding25 工具 |
| `scripts/data_preparation/spike/prepare_spike_tfp.py` | TFP artifact 预计算 |
| `scripts/data_preparation/spike/prepare_spike_raw_window.py` | raw-window artifact 预计算 |
| `scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py` | SCFlow encoding25 artifact 预计算 |
| `models/architectures/vrt/vrt.py` | 输入策略、fusion 调度、光流调度、SGP 对齐 |
| `models/fusion/` | fusion adapter/operator/reducer |
| `models/optical_flow/` | SpyNet、SCFlow 等光流后端 |
