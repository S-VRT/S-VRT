# SCFlow 子帧光流与 RGB+Spike Fusion

本文档描述当前 SCFlow 与 RGB+Spike early fusion 的集成方式。SCFlow 使用独立的 `encoding25` Spike flow 表征，不直接消费 restoration Spike TFP/raw-window 通道。

## 1. 背景

RGB+Spike fusion 有两条相关但独立的数据路径：

1. Restoration Spike 路径：用于图像恢复，返回 `L_spike=[T,S,H,W]`。
2. SCFlow Spike 路径：用于光流估计，返回 `L_flow_spike=[T*S,25,H,W]`。

其中 `S` 是 `spike_channels`，也是 `spike_flow.subframes`。在 early fusion 展开或 collapsed 子帧组合时，这两个值必须一致。

## 2. Artifact 格式

每个 RGB frame 对应一个 `.dat` 文件：

```text
<dataroot_spike>/<clip>/spike/<frame>.dat
```

SCFlow 需要从 `.dat` 中提取以某个中心为基准的 25-bin Spike window：

```text
single frame: [25,H,W]
subframes:    [S,25,H,W]
```

输出目录：

```text
S=1:
  <clip>/encoding25_dt10/<frame>.npy

S>1:
  <clip>/encoding25_dt10_s<S>/<frame>.npy
```

示例：

```text
GOPR0384_11_02/
  spike/
    001301.dat
  encoding25_dt10_s4/
    001301.npy        # [4,25,360,640]
```

## 3. 子中心选取

对每个 `.dat` 的局部时间轴，在有效范围 `[12, T_raw - 13]` 内均匀选取 `S` 个中心：

```text
T_raw=56, S=4 -> [12,22,33,43]
T_raw=88, S=4 -> [12,33,54,75]
```

每个中心提取一个 `25` bin window。`S=1` 时退化为单个中心 window。

## 4. Dataset 契约

配置：

```json
"spike_flow": {
  "representation": "encoding25",
  "dt": 10,
  "root": "auto",
  "subframes": 4,
  "collapse_policy": "mean_windows"
}
```

字段说明：

| 字段 | 默认 | 说明 |
|------|------|------|
| `representation` | `""` | SCFlow 模式必须为 `encoding25` |
| `dt` | `10` | artifact 目录名的一部分 |
| `root` | `auto` | `auto` 表示跟随 `dataroot_spike` |
| `subframes` | `1` | 每个 RGB frame 的 SCFlow 子窗口数 |
| `collapse_policy` | `mean_windows` | collapsed fusion 下的子帧处理策略 |

Dataset 返回：

```text
sample["L_flow_spike"]: [T*S,25,H,W]
```

DataLoader 后：

```text
L_flow_spike: [B,T*S,25,H,W]
```

关键约束：

```text
spike_flow.subframes == spike_channels
```

当 `subframes > 1` 且二者不等时，Dataset 初始化会报错。

## 5. VRT 光流路径

SCFlow 由 `netG.optical_flow.module="scflow"` 启用：

```json
"netG": {
  "optical_flow": {
    "module": "scflow",
    "checkpoint": "weights/optical_flow/dt10_e40.pth",
    "params": {}
  }
}
```

`models/architectures/vrt/vrt.py` 中：

1. `forward()` 接收 `flow_spike`。
2. early fusion 产生 `backbone_view` 和 fusion metadata。
3. `_align_flow_spike_to_fused_time_axis()` 或 `_compose_subframe_flow_sequence()` 对齐 SCFlow 时间轴。
4. `get_flow_2frames()` 调用 SCFlow wrapper。
5. `get_aligned_image_2frames()` 使用 frame-level flow 做 SGP/parallel warping。

SCFlow 输入校验：

- `flow_spike.ndim == 5`
- channel 维必须是 `25`
- spatial size 必须等于当前 backbone `H,W`
- temporal size 必须匹配当前 backbone 时间轴，或在 `compose_subframes` 下匹配 `T*S`

## 6. Expanded Frame Contract

Expanded early fusion 中，主干时间轴已经是子帧级：

```text
RGB:        [B,T,3,H,W]
Spike:      [B,T,S,H,W]
backbone:   [B,T*S,3,H,W]
flow_spike: [B,T*S,25,H,W]
```

此时 SCFlow 直接在 `T*S` 时间轴上估计相邻子帧 flow。Restoration 输出再根据 reducer 回到 `[B,T,3,H,W]`，或在 interpolation 模式保留展开时间轴。

## 7. Collapsed Frame Contract

Collapsed early fusion 中，主干保持 RGB 帧级：

```text
RGB:        [B,T,3,H,W]
Spike:      [B,T,S,H,W]
backbone:   [B,T,3,H,W]
flow_spike: [B,T*S,25,H,W]
```

这时 `spike_flow.collapse_policy` 决定如何把子帧 SCFlow 信息用于 RGB 帧级主干。

### 7.1 `mean_windows`

默认策略。VRT 将：

```text
[B,T*S,25,H,W] -> reshape [B,T,S,25,H,W] -> mean over S -> [B,T,25,H,W]
```

优点：

- 向后兼容。
- 计算量较低。
- shape 与旧 SCFlow 路径一致。

缺点：

- 会丢掉子帧间的细粒度运动顺序。

### 7.2 `compose_subframes`

新策略。VRT 保留 `T*S` flow Spike 输入：

1. 在相邻子帧 window 上运行 SCFlow，得到 `[B,T*S-1,2,H,W]`。
2. 选每个 RGB frame 的 anchor 子帧，`anchor = S // 2`。
3. 将 `t*S+anchor` 到 `(t+1)*S+anchor` 的相邻 flow 逐段 compose。
4. 返回 RGB 帧级 flow `[B,T-1,2,H,W]`。

组合公式遵循 VRT 现有 flow convention：

```python
composed = flow_a_to_b + flow_warp(flow_b_to_c, flow_a_to_b.permute(0, 2, 3, 1))
```

优点：

- 保留子帧运动顺序。
- 主干仍保持 collapsed 的 `[B,T,3,H,W]` 计算量。

约束：

- `spike_bins > 1`
- `flow_spike.size(1) == T * spike_bins`
- train/test 的 `collapse_policy` 必须一致

## 8. 数据准备命令

生成 S=4 SCFlow artifact：

```bash
uv run python scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --dt 10 \
  --subframes 4 \
  --spike-h 360 \
  --spike-w 640 \
  --num-workers 16
```

生成 raw-window21 对应的 S=21 SCFlow artifact：

```bash
uv run python scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --dt 10 \
  --subframes 21 \
  --spike-h 360 \
  --spike-w 640 \
  --num-workers 16
```

先估算空间：

```bash
uv run python scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --dt 10 \
  --subframes 4 \
  --space-only
```

Dry run：

```bash
uv run python scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --dt 10 \
  --subframes 4 \
  --dry-run
```

## 9. 配置示例

Collapsed PASE residual + mean windows：

```json
{
  "datasets": {
    "train": {
      "spike_channels": 4,
      "spike_flow": {
        "representation": "encoding25",
        "dt": 10,
        "root": "auto",
        "subframes": 4,
        "collapse_policy": "mean_windows"
      }
    }
  },
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
      "out_chans": 3,
      "early": {
        "frame_contract": "collapsed"
      }
    },
    "optical_flow": {
      "module": "scflow",
      "checkpoint": "weights/optical_flow/dt10_e40.pth"
    }
  }
}
```

Collapsed Mamba + composed subframes：

```json
{
  "datasets": {
    "train": {
      "spike_channels": 9,
      "spike_flow": {
        "representation": "encoding25",
        "dt": 10,
        "root": "auto",
        "subframes": 9,
        "collapse_policy": "compose_subframes"
      }
    },
    "test": {
      "spike_channels": 9,
      "spike_flow": {
        "representation": "encoding25",
        "dt": 10,
        "root": "auto",
        "subframes": 9,
        "collapse_policy": "compose_subframes"
      }
    }
  }
}
```

## 10. 测试

SCFlow contract 和 artifact 工具：

```bash
uv run pytest tests/models/test_optical_flow_scflow_contract.py \
              tests/models/test_optical_flow_scflow_integration.py -v
```

VRT fusion + SCFlow collapse policy：

```bash
uv run pytest tests/models/test_vrt_fusion_integration.py -v
```

相关测试覆盖：

| 测试 | 覆盖点 |
|------|--------|
| `test_compute_subframe_centers_*` | 子中心选择 |
| `test_validate_subframes_tensor_*` | `[S,25,H,W]` shape 校验 |
| `test_dataset_parses_spike_flow_subframes` | Dataset 配置解析 |
| `test_dataset_load_subframe_flow_spike` | 子帧 artifact 加载 |
| `test_vrt_parses_scflow_collapse_policy_from_dataset_config` | VRT 解析 collapse policy |
| `test_vrt_get_flow_2frames_compose_subframes_returns_frame_level_flows` | compose 后 frame-level flow shape |
| `test_vrt_structured_early_mamba_mean_windows_collapses_subframe_flow_spike` | collapsed 默认均值路径 |
| `test_vrt_structured_early_mamba_compose_subframes_keeps_expanded_flow_spike` | compose 保留展开 flow Spike |

## 11. 常见问题

### `subframes` 必须等于 `spike_channels` 吗？

在 `subframes > 1` 时必须相等。Restoration Spike 和 SCFlow Spike 需要共享同一子帧倍率，否则 early fusion 时间轴无法可靠对齐。

### 不启用 SCFlow 怎么办？

使用 `netG.optical_flow.module="spynet"`，不配置 `spike_flow` 或让它不产生 `L_flow_spike`。SpyNet 只使用 RGB 通道。

### 不启用 early fusion 怎么办？

concat + SpyNet 路径不需要 `L_flow_spike`。若 concat + SCFlow，需要确保 `flow_spike` temporal length 与 VRT 当前 `T` 匹配。

### S=21 是否合理？

S=21 通常对应 raw-window21 restoration 表征。它可以用于 SCFlow 子帧 artifact，但计算量和磁盘占用会明显上升。确认空间后再生成。

### SCFlow 权重会训练吗？

当前 SCFlow wrapper 按推理路径使用，训练主线不依赖解冻 SCFlow。若未来需要端到端 finetune，应单独设计 optimizer、显存和稳定性策略。

## 12. 排查顺序

1. 确认 `L_flow_spike` 是否存在。
2. 打印 `L_flow_spike.shape`，应为 `[B,T*S,25,H,W]`。
3. 检查 artifact 目录名是否与 `dt`/`subframes` 一致。
4. 检查 `spike_channels == spike_flow.subframes`。
5. 检查 train/test 的 `collapse_policy` 是否一致。
6. 检查 `fusion.early.frame_contract` 是 `expanded` 还是 `collapsed`。
7. 若 spatial mismatch，检查 Dataset crop/resize 后的 `H,W` 是否与 backbone 输入一致。
