# SCFlow Subframe Encoding25 Design

**Date:** 2026-04-16
**Status:** Implemented
**Depends on:** SCFlow Strict Semantic Integration (2026-04-13), Early Fusion Temporal Expansion (2026-04-14)

---

## 1. Problem

Early fusion temporal expansion 把 VRT 主干时间轴从 N 帧展开到 N×S 帧（S = `spike_channels`）。但 SCFlow 的 `flow_spike` 仍然是 `[B, N, 25, H, W]`，导致 `get_flow_2frames()` 中 `flow_spike.size(1) != x.size(1)` 校验失败。

两个系统各自的设计都是正确的：
- SCFlow 按 strict semantic 设计，吃原始帧级的 `flow_spike`
- Early fusion 把 T×S 展开以捕获细粒度时间信息

但它们之间缺少时间轴协调——early fusion 改了主干时间维度后，flow 路径还在用原始 N 帧粒度。

## 2. Design

### 2.1 Core Idea

为每个 `.dat` 文件生成 S 个子帧 encoding25 窗口（而非 1 个），使 `flow_spike` 的时间维度变成 `frames × S`，与 early fusion 展开后的主干时间轴对齐。

### 2.2 Data Model

每个视频帧对应一个独立的 `.dat` 文件，raw spike 时间长度 `T_raw` 不固定（观测到 56 和 88）。

**子中心选取策略**：基于每个 `.dat` 自己的 `T_raw`，不依赖全局 `center_offset + k × dt` 公式。

```
有效中心范围: [12, T_raw - 13]
  （25-bin 窗口需要 ±12 的 margin）

S 个子中心: linspace(12, T_raw - 13, S) 取整

例:
  T_raw=56, S=4: centers = [12, 22, 33, 43], sub_dt ≈ 10.3
  T_raw=88, S=4: centers = [12, 33, 54, 75], sub_dt ≈ 21
  T_raw=88, S=8: centers = [12, 21, 30, 39, 48, 57, 66, 75], sub_dt ≈ 9
```

每个子中心提取一个 25-bin 窗口，最终输出 `[S, 25, H, W]` per frame。

### 2.3 Shape Flow (端到端)

```
Dataset 层:
  per .dat:  [T_raw, H, W] → compute_subframe_centers(T_raw, S)
                            → S × build_centered_window()
                            → [S, 25, H, W]  (存为 .npy)

  per clip:  加载 N 帧 × [S, 25, H, W]
             crop/augment 每个子窗口独立处理
             展平为 [N×S, 25, H, W]

DataLoader batch:
  L_flow_spike: [B, N×S, 25, H, W]

VRT forward:
  early fusion: x [B, N, C, H, W] → [B, N×S, 3, H, W]
  get_flow_2frames():
    flow_spike.size(1) == x.size(1) == N×S  ✓
    相邻子帧对 → SCFlow → (N×S - 1) 对 flow
```

### 2.4 Key Constraint: S == spike_channels

`spike_flow.subframes` (S) **必须**等于 `spike_channels`（early fusion 的展开因子）。
两者不等时 dataset `__init__()` 会 raise ValueError。

原因：early fusion 把时间轴展开 `spike_channels` 倍，flow 路径也必须提供同样的时间粒度。

### 2.5 SCFlow Training Domain

SCFlow 训练时使用 dt=10 或 dt=20 的帧对间距。子帧间距 `sub_dt` 应尽量接近训练域：

| T_raw | S | sub_dt | 训练域 | 评价 |
|-------|---|--------|--------|------|
| 56 | 4 | ~10.3 | dt=10 | 很好 |
| 56 | 8 | ~4.4 | — | 偏离，需 finetune |
| 88 | 4 | ~21 | dt=20 | 可接受 |
| 88 | 8 | ~9 | dt=10 | 很好 |

**推荐 S=4 作为第一版**，两种 T_raw 都落在训练域内。SCFlow 权重冻结。

## 3. Configuration

### 3.1 spike_flow 配置

```json
"spike_flow": {
    "representation": "encoding25",
    "dt": 10,
    "root": "auto",
    "subframes": 4
}
```

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `representation` | str | `""` | 必须为 `"encoding25"` when `module=scflow` |
| `dt` | int | `10` | SCFlow 训练参数，进入目录名和 wrapper dt |
| `root` | str | `"auto"` | `"auto"` 跟随 `dataroot_spike`，或指定绝对路径 |
| `subframes` | int | `1` | 每帧子窗口数。>1 时必须等于 `spike_channels` |

### 3.2 Artifact 目录

- S=1（向后兼容）: `<spike_root>/<clip>/encoding25_dt{dt}/<frame>.npy`，shape `[25, H, W]`
- S>1: `<spike_root>/<clip>/encoding25_dt{dt}_s{S}/<frame>.npy`，shape `[S, 25, H, W]`

### 3.3 Preparation Script

```bash
python -m scripts.data_preparation.spike_flow.prepare_scflow_encoding25 \
    --spike-root <dataroot_spike> \
    --dt 10 \
    --subframes 4 \
    --spike-h 360 --spike-w 640
```

新增参数 `--subframes`（default 4）。

### 3.4 完整配置示例

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
    },
    "train": {
        "freeze_backbone": true
    }
}
```

注意：`spike_channels` 和 `spike_flow.subframes` 都设为 4。

## 4. Implementation

### 4.1 Module: `data/spike_recc/encoding25.py`

新增函数：

- `compute_subframe_centers(t_raw, num_subframes, margin=12)` — 在 `.dat` 局部时间轴上选 S 个等距中心
- `validate_subframes_tensor(tensor, num_subframes)` — 校验 `[S, 25, H, W]`
- `build_output_dir_subframes(clip_dir, dt, num_subframes)` — 构建带 `_s{S}` 后缀的目录路径

### 4.2 Script: `prepare_scflow_encoding25.py`

新增函数：

- `build_scflow_subframe_windows(spike_matrix, num_subframes)` — 从一个 `.dat` 提取 S 个 25-bin 窗口

修改 `process_clip()` 接受 `num_subframes` 参数，`main()` 接受 `--subframes` CLI 参数。

### 4.3 Dataset: `dataset_video_train_rgbspike.py`

- `_parse_spike_flow_config()` 解析 `spike_flow.subframes`
- `__init__()` 校验 `subframes == spike_channels`（当 subframes > 1）
- `_load_encoded_flow_spike()` 根据 subframes 选择目录名和 shape 校验
- `__getitem__()` 对每个子窗口独立 crop/resize，展平后进入 augment 流水线

### 4.4 Model/VRT

- `model_plain.py`: 不改逻辑，更新注释说明 `L_flow_spike.size(1)` 可能是 `frames × subframes`
- `vrt.py` `get_flow_2frames()`: 改进 error message，提示检查 `spike_flow.subframes` 和 `spike_channels` 是否一致

## 5. Relation to Prior Specs

### 5.1 SCFlow Strict Semantic Integration (2026-04-13)

该 spec 定义了 `L_flow_spike` 的原始契约 `[B, T, 25, H, W]`。本设计是其扩展：

- **不变**: encoding25 的 25-channel 语义、SCFlow wrapper 校验、`representation=encoding25` 要求
- **扩展**: `T` 从"视频帧数 N"变为"N × subframes"；artifact 目录加 `_s{S}` 后缀
- **废弃**: 全局 `center_offset + k × dt` 索引方案（因每个 `.dat` 是独立文件，不适用全局公式）

### 5.2 Early Fusion Temporal Expansion (2026-04-14)

该 spec 定义了 `[B, N, C, H, W] → [B, N×S, 3, H, W]` 的展开。本设计补充了 flow 路径：

- **不变**: SpikeUpsample、temporal expansion、restoration reducer
- **补充**: flow_spike 也按 N×S 粒度提供，使 `get_flow_2frames()` 的时间维度校验自然通过
- **约束**: `spike_flow.subframes` 必须等于 `spike_channels`

## 6. Future Work

- S=8 支持：需要验证 T_raw=56 时 sub_dt≈4.4 的 SCFlow 精度，可能需要 finetune
- SCFlow 解冻：在子帧 flow 稳定后考虑解冻权重做端到端训练
- Flow 预计算：将 SCFlow inference 移到离线预处理，减少训练时计算开销
