# VRT+Spike 架构快速参考

快速查阅模型架构、维度和配置的簡明指南。

---

## 📊 一图概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      VRT+Spike 视频去模糊架构                              │
└─────────────────────────────────────────────────────────────────────────┘

输入层 INPUTS
═══════════════════════════════════════════════════════════════════════════
  RGB 模糊帧                          Spike 体素化数据
  (B, T, 3, H, W)                    (B, T, K, H, W)
  T=5, H=W=256                       K=32 bins
       │                                    │
       │                                    │
       ▼                                    ▼
───────────────────────────────────────────────────────────────────────────

编码层 ENCODING (并行处理)
═══════════════════════════════════════════════════════════════════════════
┌──────────────────────────┐      ┌──────────────────────────────────────┐
│   VRT RGB 编码器          │      │   Spike 编码器 (SpikeEncoder3D)       │
│   + TMSA 时序建模         │      │   3D 卷积金字塔                       │
├──────────────────────────┤      ├──────────────────────────────────────┤
│ Stage 1: (B,96,T,H,W)    │      │ Permute: (B,T,K,H,W)→(B,K,T,H,W)     │
│ Stage 2: (B,96,T',H/2..) │      │ in_proj: Conv3d(32→96)               │
│ Stage 3: (B,96,T'',H/4..)│      │ ┌─────────────────────────────────┐  │
│ Stage 4: (B,96,T''',H/8.│      │ │ Scale 1: (B,96,T,H,W)   ←res0  │  │
│          时间逐渐压缩     │      │ │ Scale 2: (B,96,T,H/2..) ←res1  │  │
└──────────────────────────┘      │ │ Scale 3: (B,96,T,H/4..) ←res2  │  │
       │                          │ │ Scale 4: (B,96,T,H/8..) ←res3  │  │
       │ Fr₁..₄ (4个尺度)          │ └─────────────────────────────────┘  │
       │                          │          │ Fs₁..₄                    │
       │                          │          ▼                            │
       │                          │ ┌─────────────────────────────────┐  │
       │                          │ │  SpikeTemporalSA 时序自注意力    │  │
       │                          │ │  沿时间维度 Self-Attention       │  │
       │                          │ │  Fs₁..₄ → Fs'₁..₄               │  │
       │                          │ └─────────────────────────────────┘  │
       │                          └──────────────────────────────────────┘
       │                                    │ Fs'₁..₄
───────────────────────────────────────────────────────────────────────────

融合层 FUSION (Cross-Attention)
═══════════════════════════════════════════════════════════════════════════
       │                                    │
       ▼                                    ▼
   ┌─────────────────────────────────────────────┐
   │  Multi-Scale Cross-Attention Fusion         │
   │  在每个 VRT Stage 之后注入                   │
   ├─────────────────────────────────────────────┤
   │  Scale 1: CrossAttn(Q=Fr₁, K/V=Fs'₁) → Ff₁ │
   │  Scale 2: CrossAttn(Q=Fr₂, K/V=Fs'₂) → Ff₂ │
   │  Scale 3: CrossAttn(Q=Fr₃, K/V=Fs'₃) → Ff₃ │
   │  Scale 4: CrossAttn(Q=Fr₄, K/V=Fs'₄) → Ff₄ │
   │                                             │
   │  机制: RGB 特征(Q) 查询 Spike 特征(K/V)     │
   │  学习: 动态跨模态信息交互                    │
   └─────────────────────────────────────────────┘
                      │ Ff₁..₄
───────────────────────────────────────────────────────────────────────────

解码层 DECODING
═══════════════════════════════════════════════════════════════════════════
                      ▼
   ┌─────────────────────────────────────────────┐
   │           VRT 解码器                         │
   │  Ff₄ (1/8x) → 上采样 → Ff₃ (1/4x)           │
   │          → 上采样 → Ff₂ (1/2x)               │
   │          → 上采样 → Ff₁ (1x)                 │
   │          → 重建                               │
   └─────────────────────────────────────────────┘
                      │
───────────────────────────────────────────────────────────────────────────

输出层 OUTPUT
═══════════════════════════════════════════════════════════════════════════
                      ▼
               清晰输出帧
             (B, T, 3, H, W)
             ✨ Deblurred Video
```

---

## 🔢 维度速查表

### 输入维度

| 数据 | 形状 | 示例值 | 说明 |
|------|------|--------|------|
| RGB 模糊帧 | `(B, T, 3, H, W)` | `(2, 5, 3, 256, 256)` | B=batch, T=时间帧, H×W=分辨率 |
| Spike 体素 | `(B, T, K, H, W)` | `(2, 5, 32, 256, 256)` | K=32 时间bins |

### Spike 编码器维度变化

| 阶段 | 操作 | 输入维度 | 输出维度 |
|------|------|---------|---------|
| 1. Permute | 维度转换 | `(B, T, K, H, W)` | `(B, K, T, H, W)` |
| 2. in_proj | Conv3d(32→96) | `(B, 32, T, H, W)` | `(B, 96, T, H, W)` |
| 3. res0 | 2×ResBlock3D | `(B, 96, T, H, W)` | `(B, 96, T, H, W)` ← **Fs₁** |
| 4. down1 | Conv3d stride(1,2,2) | `(B, 96, T, H, W)` | `(B, 96, T, H/2, W/2)` |
| 5. res1 | 2×ResBlock3D | `(B, 96, T, H/2, W/2)` | `(B, 96, T, H/2, W/2)` ← **Fs₂** |
| 6. down2 | Conv3d stride(1,2,2) | `(B, 96, T, H/2, W/2)` | `(B, 96, T, H/4, W/4)` |
| 7. res2 | 2×ResBlock3D | `(B, 96, T, H/4, W/4)` | `(B, 96, T, H/4, W/4)` ← **Fs₃** |
| 8. down3 | Conv3d stride(1,2,2) | `(B, 96, T, H/4, W/4)` | `(B, 96, T, H/8, W/8)` |
| 9. res3 | 2×ResBlock3D | `(B, 96, T, H/8, W/8)` | `(B, 96, T, H/8, W/8)` ← **Fs₄** |

### 多尺度特征对比

| 尺度 | VRT RGB 特征 (Fr) | Spike 特征 (Fs) | 时序增强 (Fs') | 融合特征 (Ff) |
|------|-------------------|----------------|---------------|---------------|
| **Scale 1** | `(B, 96, T, H, W)` | `(B, 96, T, H, W)` | `(B, 96, T, H, W)` | `(B, 96, T, H, W)` |
| **Scale 2** | `(B, 96, T', H/2, W/2)` | `(B, 96, T, H/2, W/2)` | `(B, 96, T, H/2, W/2)` | `(B, 96, T', H/2, W/2)` |
| **Scale 3** | `(B, 96, T'', H/4, W/4)` | `(B, 96, T, H/4, W/4)` | `(B, 96, T, H/4, W/4)` | `(B, 96, T'', H/4, W/4)` |
| **Scale 4** | `(B, 96, T''', H/8, W/8)` | `(B, 96, T, H/8, W/8)` | `(B, 96, T, H/8, W/8)` | `(B, 96, T''', H/8, W/8)` |

**注意**：
- VRT 的时间维度逐渐压缩（T → T' → T'' → T'''）
- Spike 特征保持时间维度（T 不变）
- 融合后的特征繼承 VRT 的时间维度

---

## ⚙️ 配置参数速查

### 数据配置 (DATA)

```yaml
CLIP_LEN: 5              # 时间帧数 T
CROP_SIZE: 256           # 空间裁剪大小 H×W
K: 32                    # Spike 体素化 bins
NUM_VOXEL_BINS: 32       # 同上（向后兼容）
SPIKE_DIR: spike         # Spike 数据目录
VOXEL_CACHE_DIRNAME: spike_vox  # 体素緩存目录
```

### 模型配置 (MODEL)

```yaml
USE_SPIKE: true          # 使用 Spike 集成

CHANNELS_PER_SCALE:      # 各尺度通道数（必須4个值）
  - 96
  - 96
  - 96
  - 96

VRT:
  USE_CHECKPOINT_ATTN: true   # VRT 注意力层梯度检查点
  USE_CHECKPOINT_FFN: true    # VRT FFN 层梯度检查点

SPIKE_ENCODER:           # Spike 编码器（可选）
  TEMPORAL_STRIDES: [1, 1, 1]  # 默认：时间維不下采样
  SPATIAL_STRIDES: [2, 2, 2]   # 默认：空间維下采样2倍

SPIKE_TSA:               # Spike 时序自注意力
  HEADS: 4
  DROPOUT: 0.0
  MLP_RATIO: 2
  ADAPTIVE_CHUNK: true   # 自适应分块
  MAX_BATCH_TOKENS: 49152
  CHUNK_SIZE: 64
  CHUNK_SHAPE: "square"

FUSE:                    # 跨模态融合
  TYPE: TemporalCrossAttn
  HEADS: 4
  DROPOUT: 0.0
  MLP_RATIO: 2
  ADAPTIVE_CHUNK: true
  MAX_BATCH_TOKENS: 49152
  CHUNK_SIZE: 64
  CHUNK_SHAPE: "square"
```

### 训练配置 (TRAIN)

```yaml
BATCH_SIZE: 2            # 批次大小
NUM_WORKERS: 4           # 数据加載器工作进程
EPOCHS: 100              # 训练輪数
LR: 0.0001               # 学习率
WEIGHT_DECAY: 0.0        # 權重衰減
MIXED_PRECISION: true    # 混合精度训练
ACCUMULATE_GRAD: 1       # 梯度累积步数
```

---

## 🧮 计算复杂度估算

### 参数量

假设配置：T=5, H=W=256, C=96, K=32

| 模块 | 参数量估算 | 说明 |
|------|-----------|------|
| **SpikeEncoder3D** | | |
| - in_proj | 32×96×3³ ≈ 83K | Conv3d(32→96, k=3) |
| - ResBlock3D (×8) | 96²×3³×2×8 ≈ 40M | 4个尺度，每个2个ResBlock |
| **SpikeTemporalSA** | 96²×4×4 ≈ 1.5M | 4个尺度，4头注意力 |
| **Fusion** | 96²×4×4 ≈ 1.5M | 4个尺度，4头交叉注意力 |
| **VRT** | ~20M | 原始VRT参数 |
| **總計** | ~63M | 近似值 |

### 显存占用

單个樣本（B=1, T=5, H=W=256, C=96, K=32）：

| 阶段 | 显存估算 | 主要张量 |
|------|---------|---------|
| **输入** | ~20 MB | RGB(5×3×256²) + Spike(5×32×256²) |
| **Spike编码** | ~150 MB | 4个尺度特征，各有時空维度 |
| **VRT编码** | ~100 MB | 4个尺度特征 |
| **注意力激活** | ~200 MB | SpikeTemporalSA + CrossAttn |
| **峰值** | ~500 MB | 未使用梯度检查点 |
| **优化后** | ~250 MB | 使用梯度检查点 + 自适应分块 |

**实際显存需求**（训练）：
- Batch size 2: ~8 GB（使用所有优化）
- Batch size 4: ~14 GB
- Batch size 8: ~26 GB

---

## 🔧 常见配置场景

### 场景 1: 高质量训练（充足显存）

```yaml
DATA:
  CLIP_LEN: 7              # 更多时间帧
  CROP_SIZE: 384           # 更大分辨率

TRAIN:
  BATCH_SIZE: 4
  MIXED_PRECISION: true

MODEL:
  VRT:
    USE_CHECKPOINT_ATTN: false  # 不使用检查点
    USE_CHECKPOINT_FFN: false
  
  SPIKE_TSA:
    ADAPTIVE_CHUNK: false       # 不使用分块
  
  FUSE:
    ADAPTIVE_CHUNK: false
```

### 场景 2: 低显存训练（<8GB）

```yaml
DATA:
  CLIP_LEN: 3              # 更少时间帧
  CROP_SIZE: 128           # 更小分辨率

TRAIN:
  BATCH_SIZE: 1
  ACCUMULATE_GRAD: 4       # 梯度累积模擬batch 4
  MIXED_PRECISION: true

MODEL:
  VRT:
    USE_CHECKPOINT_ATTN: true
    USE_CHECKPOINT_FFN: true
  
  SPIKE_TSA:
    ADAPTIVE_CHUNK: true
    MAX_BATCH_TOKENS: 24576   # 更小的块
    CHUNK_SIZE: 32
  
  FUSE:
    ADAPTIVE_CHUNK: true
    MAX_BATCH_TOKENS: 24576
    CHUNK_SIZE: 32
```

### 场景 3: 快速实验（速度优先）

```yaml
DATA:
  CLIP_LEN: 5
  CROP_SIZE: 256

TRAIN:
  BATCH_SIZE: 2
  NUM_WORKERS: 8           # 更多数据加載线程
  MIXED_PRECISION: true

MODEL:
  CHANNELS_PER_SCALE: [64, 64, 64, 64]  # 更少通道
  
  SPIKE_TSA:
    HEADS: 2               # 更少注意力头
    MLP_RATIO: 1
  
  FUSE:
    HEADS: 2
    MLP_RATIO: 1
```

---

## 📝 关键代碼位置

### 主要模块

```
src/models/
├── integrate_vrt.py              # VRTWithSpike 主模型
│   └── VRTWithSpike              # 整合所有组件
│       ├── __init__()            # 初始化各模块
│       ├── forward()             # 前向傳播
│       ├── _monkeypatch_forward_features()  # 注入融合层
│       └── _restore_forward_features()      # 恢复原始VRT
│
├── spike_encoder3d.py            # Spike 编码器
│   ├── ResidualBlock3D           # 3D 残差块
│   └── SpikeEncoder3D            # 多尺度编码
│       ├── in_proj               # 通道投影 32→96
│       ├── downs                 # 下采样层
│       └── residuals             # 残差块
│
├── spike_temporal_sa.py          # Spike 时序自注意力
│   └── SpikeTemporalSA           # 处理时间维度
│       ├── sa_blocks             # 各尺度的SA模块
│       └── forward()             # 并行处理4个尺度
│
└── fusion/
    └── cross_attn_temporal.py    # 跨模态融合
        └── MultiScaleTemporalCrossAttnFuse
            ├── cross_attn_blocks # 各尺度的CrossAttn
            └── forward()         # 融合Fr和Fs'
```

### 关键函数簽名

```python
# VRTWithSpike
def forward(
    self, 
    rgb_clip: Tensor,      # (B, T, 3, H, W)
    spike_vox: Tensor      # (B, T, K, H, W)
) -> Tensor:               # (B, T, 3, H, W)

# SpikeEncoder3D
def forward(
    self, 
    x: Tensor              # (B, T, K, H, W)
) -> List[Tensor]:         # [(B,C,T,H,W), (B,C,T,H/2,W/2), ...]

# SpikeTemporalSA
def forward(
    self, 
    spike_feats: List[Tensor]   # [(B,C,T,H_i,W_i), ...]
) -> List[Tensor]:              # 相同形状

# MultiScaleTemporalCrossAttnFuse
def __call__(
    self,
    rgb_feat: Tensor,      # (B, C, T', H_i, W_i)
    spike_feat: Tensor,    # (B, C, T, H_i, W_i)
    scale_idx: int         # 0..3
) -> Tensor:               # (B, C, T', H_i, W_i)
```

---

## 🐛 常见问题排查

### 维度不匹配错误

```python
RuntimeError: shape mismatch in attention

可能原因：
1. VRT 和 Spike 的 CHANNELS_PER_SCALE 不一致
2. 空间分辨率对齐问题（检查 spatial_strides）

解決方法：
- 确保配置文件中 CHANNELS_PER_SCALE 有4个相同值
- 检查输入图像尺寸是否是 2^n 倍（便于下采样）
```

### 显存溢出 (OOM)

```python
RuntimeError: CUDA out of memory

解決方法（按优先级）：
1. 啟用梯度检查点: USE_CHECKPOINT_ATTN/FFN = true
2. 啟用自适应分块: ADAPTIVE_CHUNK = true
3. 減小 BATCH_SIZE
4. 減小 CROP_SIZE
5. 減小 CLIP_LEN
6. 減小 CHANNELS_PER_SCALE (例如 [64,64,64,64])
7. 使用梯度累积: ACCUMULATE_GRAD > 1
```

### Spike 数据加載错误

```python
FileNotFoundError: spike/*.dat not found

检查清單：
1. 数据目录结构是否正确
2. SPIKE_DIR 配置是否正确
3. 是否需要預生成体素緩存
4. .dat 文件格式是否正确（uint8）
```

### 训练不收斂

```python
Loss 不下降或 NaN

排查步驟：
1. 检查学习率（嘗試降低到 1e-5）
2. 检查梯度裁剪（添加 gradient clipping）
3. 检查数据歸一化（RGB 和 Spike 都应歸一化）
4. 嘗試从預训练 VRT 權重开始
5. 检查損失函数配置
```

---

## 📚 延伸阅读

- **详细架构**: [`docs/architecture_dataflow.md`](architecture_dataflow.md)
- **可视化图表**: [`docs/architecture_diagrams.md`](architecture_diagrams.md)
- **变更总结**: [`docs/CHANGES_SUMMARY.md`](CHANGES_SUMMARY.md)
- **验证报告**: [`docs/CODE_VERIFICATION_REPORT.md`](CODE_VERIFICATION_REPORT.md)

---

**最后更新**: 2025-10-20  
**版本**: v2.0  
**維护者**: AI Assistant





