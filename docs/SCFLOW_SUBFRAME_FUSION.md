# SCFlow 子帧光流与 Early Fusion 集成

## 概述

本文档描述 SCFlow 光流模块如何与 Early Fusion 时间展开机制集成，使光流在子帧粒度上提供运动信息。

### 背景

S-VRT 中有两条独立的时间轴扩展路径：

1. **Early Fusion**：将 RGB 复制 S 份对齐 Spike 子帧，主干从 N 帧扩展到 N×S 帧
2. **SCFlow 光流**：从 Spike 原始流中提取 25-bin 时间窗口，计算相邻帧间光流

问题：early fusion 扩展主干时间轴后，光流仍在原始 N 帧粒度上工作，两者时间维度不匹配。

### 解决方案

为每个 `.dat` 文件生成 S 个子帧 encoding25 窗口，使 `flow_spike` 形状从 `[B, N, 25, H, W]` 变为 `[B, N×S, 25, H, W]`，与主干时间轴对齐。

---

## 数据模型

### 每帧 `.dat` 文件

每个视频帧对应一个独立的 `.dat` 文件，包含该帧周围的原始 spike 流：

```
<dataroot_spike>/<clip>/spike/000047.dat  →  (T_raw, H, W)
```

`T_raw` 不固定，已观测到 56 和 88 两种长度。

### 子中心选取

从每个 `.dat` 的有效中心范围内均匀选取 S 个点：

```
有效范围: [12, T_raw - 13]
子中心:   numpy.linspace(12, T_raw - 13, S) 取整

T_raw=56, S=4:  centers = [12, 22, 33, 43]   sub_dt ≈ 10.3
T_raw=88, S=4:  centers = [12, 33, 54, 75]   sub_dt ≈ 21
```

每个子中心提取一个 ±12 的 25-bin 窗口。最终每帧输出 `[S, 25, H, W]`。

### Shape 端到端

```
.dat 文件
  ↓ compute_subframe_centers(T_raw, S=4)
  ↓ build_centered_window() × S
  ↓ [S, 25, H, W]  .npy artifact

Dataset __getitem__()
  ↓ 加载 N 帧 × [S, 25, H, W]
  ↓ 独立 crop/resize 每个子窗口
  ↓ 展平为 [N×S, 25, H, W]
  ↓ augment (flip/rotate)
  → sample['L_flow_spike']: [N×S, 25, H, W]

DataLoader
  → L_flow_spike: [B, N×S, 25, H, W]

VRT forward()
  ↓ early fusion: x [B, N, C, H, W] → [B, N×S, 3, H, W]
  ↓ get_flow_2frames(x, flow_spike)
  ↓ flow_spike.size(1) == x.size(1) == N×S  ✓
  ↓ SCFlow(sub_i, sub_{i+1}) → (N×S - 1) 对 flow
  → flow-guided temporal attention 在子帧粒度对齐
```

---

## 配置

### 最小配置

```json
{
    "datasets": {
        "train": {
            "spike_channels": 4,
            "spike_flow": {
                "representation": "encoding25",
                "dt": 10,
                "root": "auto",
                "subframes": 4
            }
        },
        "test": {
            "spike_flow": {
                "representation": "encoding25",
                "dt": 10,
                "root": "auto",
                "subframes": 4
            }
        }
    },
    "netG": {
        "optical_flow": {
            "module": "scflow",
            "checkpoint": "weights/optical_flow/dt10_e40.pth"
        },
        "fusion": {
            "enabled": true,
            "placement": "early",
            "operator": "gated"
        }
    }
}
```

### spike_flow 配置项

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `representation` | str | `""` | SCFlow 模式下必须为 `"encoding25"` |
| `dt` | int | `10` | SCFlow 训练参数（10 或 20），进入目录名 |
| `root` | str | `"auto"` | `"auto"` 跟随 `dataroot_spike`；也可指定绝对路径 |
| `subframes` | int | `1` | 每帧子窗口数。设为 >1 时**必须等于 `spike_channels`** |

### 关键约束

**`spike_flow.subframes` 必须等于 `spike_channels`。** 两者不等时 dataset 初始化会报错：

```
ValueError: spike_flow.subframes (8) must equal spike_channels (4)
             for early-fusion temporal axis alignment.
```

原因：early fusion 按 `spike_channels` 展开时间轴，flow 路径必须按同样的倍数提供子帧光流。

---

## 数据准备

### 生成 encoding25 子帧 artifact

```bash
python -m scripts.data_preparation.spike_flow.prepare_scflow_encoding25 \
    --spike-root /path/to/GOPRO_Large_spike_seq/train \
    --dt 10 \
    --subframes 4 \
    --spike-h 360 \
    --spike-w 640 \
    --num-workers 8
```

#### 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--spike-root` | (必需) | 包含 clip 子目录的 spike 数据根目录 |
| `--dt` | `10` | SCFlow 训练间距 |
| `--subframes` | `4` | 每帧生成的子窗口数 |
| `--spike-h` / `--spike-w` | `360` / `640` | spike 相机分辨率 |
| `--num-workers` | `1` | 并行 worker 数 |
| `--dry-run` | - | 只统计不生成 |
| `--overwrite` | - | 覆盖已有文件 |

#### 输出目录结构

```
<spike_root>/
  <clip>/
    spike/
      000001.dat          ← 原始 spike 流
      000002.dat
      ...
    encoding25_dt10_s4/   ← 新生成的子帧 artifact
      000001.npy          ← shape [4, 25, 360, 640]
      000002.npy
      ...
```

S=1 时目录名为 `encoding25_dt10`（向后兼容），文件 shape 为 `[25, H, W]`。

#### 验证

```bash
# dry-run 检查
python -m scripts.data_preparation.spike_flow.prepare_scflow_encoding25 \
    --spike-root /path/to/train --subframes 4 --dry-run

# 检查单个文件
python -c "
import numpy as np
arr = np.load('path/to/encoding25_dt10_s4/000001.npy')
print(arr.shape, arr.dtype)  # 应输出 (4, 25, 360, 640) float32
"
```

---

## SCFlow 训练域

SCFlow 原始训练使用 dt=10（或 dt=20）间距的帧对。子帧间距 `sub_dt` 应尽量接近训练域：

| T_raw | S | sub_dt | 对应训练域 | 评价 |
|-------|---|--------|-----------|------|
| 56 | 4 | ~10.3 | dt=10 | ✓ 匹配 |
| 88 | 4 | ~21 | dt=20 | ✓ 可接受 |
| 56 | 8 | ~4.4 | — | ✗ 偏离，需 finetune |
| 88 | 8 | ~9 | dt=10 | ✓ 接近 |

**推荐第一版使用 S=4**。SCFlow 权重冻结（`torch.no_grad()`），不参与反向传播。

---

## 代码结构

### 核心模块

| 文件 | 职责 |
|------|------|
| `data/spike_recc/encoding25.py` | 子中心计算、窗口提取、shape 校验 |
| `scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py` | 离线 artifact 生成 |
| `data/dataset_video_train_rgbspike.py` | 加载 `[S,25,H,W]`、crop、augment、展平 |
| `models/architectures/vrt/vrt.py` | `get_flow_2frames()` 消费 `[B, N×S, 25, H, W]` |
| `models/optical_flow/scflow/wrapper.py` | SCFlow 前向，校验 `[B, 25, H, W]` 输入对 |

### 关键函数

```python
# data/spike_recc/encoding25.py

compute_subframe_centers(t_raw=56, num_subframes=4)
# → [12, 22, 33, 43]

build_centered_window(spike_matrix, center=22)
# → [25, H, W] float32

validate_subframes_tensor(arr, num_subframes=4)
# 校验 shape == [4, 25, H, W]

build_output_dir_subframes(clip_dir, dt=10, num_subframes=4)
# → <clip>/encoding25_dt10_s4
```

```python
# scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py

build_scflow_subframe_windows(spike_matrix, num_subframes=4)
# → [4, 25, H, W] float32
```

---

## 测试

```bash
# 运行所有 SCFlow 相关测试
python -m pytest tests/models/test_optical_flow_scflow_contract.py \
                 tests/models/test_optical_flow_scflow_integration.py -v
```

### 子帧相关测试覆盖

| 测试 | 验证内容 |
|------|---------|
| `test_compute_subframe_centers_t56_s4` | T=56, S=4 的子中心在 [12, 43] 内均匀分布 |
| `test_compute_subframe_centers_t88_s4` | T=88, S=4 的子中心在 [12, 75] 内均匀分布 |
| `test_compute_subframe_centers_s1_returns_midpoint` | S=1 退化为中点 |
| `test_compute_subframe_centers_rejects_too_short` | T<25 报错 |
| `test_validate_subframes_tensor_*` | shape 校验 |
| `test_build_scflow_subframe_windows_shape_*` | 完整提取流程 |
| `test_dataset_parses_spike_flow_subframes` | 配置解析 |
| `test_dataset_load_subframe_flow_spike` | 文件加载和 shape |

---

## 常见问题

### 1. `spike_flow.subframes` 和 `spike_channels` 必须相等吗？

是的。当 `subframes > 1` 时，dataset 初始化会校验两者相等。原因是 early fusion 按 `spike_channels` 倍展开时间轴，flow 路径也必须有同样的时间分辨率。

### 2. 不用 early fusion 时怎么配？

设 `subframes: 1`（默认值）即可，行为和之前完全一样。flow_spike 形状为 `[B, N, 25, H, W]`。

### 3. 可以用 S=8 吗？

可以，但需要 `spike_channels` 也设为 8。注意 T_raw=56 时 sub_dt≈4.4，偏离 SCFlow 训练域（dt=10/20），可能需要 finetune。T_raw=88 时 sub_dt≈9，接近 dt=10 训练域。

### 4. encoding25 的 25 是什么含义？

SCFlow 的输入格式：每个"帧"是原始 spike 流中以某个时刻为中心的 25 个连续 time bin（±12）。这 25 个 bin 是 SCFlow 网络的第一层卷积硬编码的输入通道数。

### 5. 全局 `center_offset + k × dt` 公式还用吗？

不用了。每个 `.dat` 是独立文件，没有跨帧连续时间轴，全局索引公式不适用。子中心完全基于每个 `.dat` 自身的 `T_raw` 在局部时间轴上选取。`encoding25.py` 中的 `compute_center_index()` 和 `validate_center_bounds()` 保留但不再被调用。

### 6. SCFlow 权重需要训练吗？

第一版冻结权重。SCFlow 在 `wrapper.py` 中使用 `torch.no_grad()` 推理。后续可以考虑解冻做端到端 finetune。
