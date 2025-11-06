# VRT (Video Restoration Transformer) 模型架构详解

## 1. 概述

VRT (Video Restoration Transformer) 是一个基于 Transformer 架构的视频恢复模型，主要用于视频去模糊 (Video Deblurring)、视频超分辨率 (Video Super-Resolution) 等任务。本文档详细介绍 VRT 的架构设计、模块组成、数据流和训练过程。

### 1.1 主要特点

- **基于 Transformer**: 使用窗口自注意力和互注意力机制处理时序信息
- **多阶段特征提取**: 7 个下采样阶段 + 6 个上采样阶段，U型结构
- **光流引导对齐**: 使用 SpyNet 预训练模型估计光流，引导特征对齐
- **可变形卷积对齐**: DCN (Deformable Convolution Network) 实现精确的特征对齐
- **支持多模态输入**: 支持 RGB (3通道) 和 RGB+Spike (4通道) 输入

### 1.2 输入输出规格

- **输入**: `(B, N, C, H, W)`
  - `B`: Batch size
  - `N`: 帧数 (通常为 6)
  - `C`: 通道数 (RGB: 3, RGB+Spike: 4)
  - `H, W`: 空间分辨率 (通常 224×224)
  
- **输出**: `(B, N, 3, H, W)` (RGB 三通道)

---

## 2. 整体架构

### 2.1 架构层次结构

```
VRT
├── 输入层 (conv_first)
│   └── 3D Conv: C*(1+2*pa_frames) → embed_dim[0]
│
├── 光流估计模块 (SpyNet)
│   └── 计算前后帧之间的光流
│
├── 特征对齐模块 (DCN)
│   └── 使用光流引导的特征对齐
│
├── 特征提取主网络 (7个 Stage)
│   ├── Stage 1-4: 下采样路径 (down)
│   ├── Stage 5-7: 上采样路径 (up)
│   └── 残差连接: Stage7→Stage6→Stage5
│
├── Stage 8 (6个 RTMSA 层)
│   └── 残差时序互自注意力
│
├── 输出层 (conv_after_body + conv_last)
│   └── embed_dim[-1] → embed_dim[0] → out_chans
│
└── 残差连接
    └── output = x + x_lq[:, :, :out_chans, :, :]
```

### 2.2 核心设计思路

1. **时序建模**: 通过窗口自注意力和互注意力捕捉时序依赖
2. **多尺度特征**: U型结构提取不同尺度的特征
3. **特征对齐**: 光流+可变形卷积实现精确的帧间对齐
4. **残差学习**: 多层次的残差连接保证梯度流畅

---

## 3. 主要模块详解

### 3.1 输入层 (conv_first)

**位置**: `network_vrt.py` 第 1297-1305 行

**功能**: 将输入的多帧图像投影到特征空间

**实现**:
```python
if self.pa_frames:
    conv_first_in_chans = in_chans*(1+2*pa_frames)  # RGB+Spike: 4*9 = 36
else:
    conv_first_in_chans = in_chans

self.conv_first = nn.Conv3d(
    conv_first_in_chans, 
    embed_dims[0], 
    kernel_size=(1, 3, 3), 
    padding=(0, 1, 1)
)
```

**张量形状变化**:
- 输入: `(B, N, C*(1+2*2), H, W)` → `(B, 9*C, N, H, W)` (转置后)
- 输出: `(B, embed_dims[0], N, H, W)`

**说明**:
- `pa_frames=2` 表示使用前后各 2 帧进行对齐
- 输入包含: 当前帧 + 前 2 帧 + 后 2 帧 → 共 9 帧的信息
- RGB+Spike (4通道) 时，输入通道数为 `4*9=36`

### 3.2 光流估计模块 (SpyNet)

**位置**: `network_vrt.py` 第 359-516 行

**功能**: 估计相邻帧之间的光流 (Optical Flow)

**关键代码** (`get_flow_2frames`):
```python
def get_flow_2frames(self, x):
    b, n, c, h, w = x.size()
    
    # RGB+Spike 输入时，仅使用 RGB 通道 (前3个通道)
    if c > 3:
        x_rgb = x[:, :, :3, :, :]  # 提取 RGB 通道
    else:
        x_rgb = x
    
    # 前向后向光流估计
    flows_backward = self.spynet(x_1, x_2)  # t→t+1
    flows_forward = self.spynet(x_2, x_1)   # t+1→t
```

**输出**: 
- `flows_backward`: 4 个尺度的后向光流 `[(B, N-1, 2, H, W), ..., (B, N-1, 2, H/8, W/8)]`
- `flows_forward`: 4 个尺度的前向光流 (同上)

**特点**:
- SpyNet 预训练在 RGB 图像上，只接受 3 通道输入
- 光流依赖 RGB 的纹理和边缘信息
- Spike 通道不参与光流计算，但在后续特征对齐中会使用

### 3.3 特征对齐模块 (DCNv2PackFlowGuided)

**位置**: `network_vrt.py` 第 267-340 行

**功能**: 使用光流引导的可变形卷积对齐特征

**实现原理**:
1. 先用光流进行粗略对齐 (flow_warp)
2. 再用可变形卷积 (DCN) 进行精细对齐

**在 Stage 中的应用**:
```python
# Stage 中的 parallel warping
x_backward, x_forward = get_aligned_feature_2frames(x, flows_backward, flows_forward)
x = self.pa_fuse(torch.cat([x, x_backward, x_forward], 2))
```

**张量形状**:
- 输入: `(B, C, N, H, W)`
- 对齐后: `(B, C*3, N, H, W)` (当前帧 + 后向对齐 + 前向对齐)
- 融合后: `(B, C, N, H, W)`

### 3.4 时序互自注意力 (TMSA)

**位置**: `network_vrt.py` 第 728-852 行

**功能**: 结合自注意力和互注意力机制建模时序关系

#### 3.4.1 WindowAttention

**位置**: `network_vrt.py` 第 588-726 行

**两种注意力机制**:

1. **自注意力 (Self-Attention)**:
   - 每个 token 关注窗口内所有 token
   - 使用相对位置编码 (Relative Position Encoding)
   - 公式: `Attention(Q, K, V) = softmax(QK^T / sqrt(d) + bias) V`

2. **互注意力 (Mutual Attention)**:
   - 将窗口分为两部分: 前 N/2 帧和后 N/2 帧
   - 前部分用后部分的 K、V，后部分用前部分的 K、V
   - 实现帧间的信息交互
   - 使用正弦位置编码

**实现流程**:
```
输入 x: (B*nW, Wd*Wh*Ww, C)
    ↓
自注意力
    ↓
互注意力 (如果 mut_attn=True)
    ↓
拼接并投影
    ↓
输出: (B*nW, Wd*Wh*Ww, C)
```

#### 3.4.2 TMSA 完整流程

```
输入: (B, C, D, H, W)
    ↓
reshape: (B, D, H, W, C)
    ↓
LayerNorm
    ↓
窗口分割: (B*nW, Wd*Wh*Ww, C)
    ↓
WindowAttention
    ↓
DropPath
    ↓
LayerNorm + MLP (GEGLU)
    ↓
DropPath
    ↓
输出: (B, C, D, H, W)
```

**关键参数**:
- `window_size`: `[6, 8, 8]` (时间, 高度, 宽度)
- `shift_size`: 窗口滑动偏移
- `mut_attn`: 是否使用互注意力

### 3.5 TMSAG (时序互自注意力组)

**位置**: `network_vrt.py` 第 855-936 行

**功能**: 多个 TMSA 块的堆叠

**结构**:
```python
self.blocks = nn.ModuleList([
    TMSA(...) for i in range(depth)
])
```

**特点**:
- 交替使用 shift 和 non-shift 窗口
- 偶数层: `shift_size=[0,0,0]` (非移位)
- 奇数层: `shift_size=window_size//2` (移位)

### 3.6 Stage (特征提取阶段)

**位置**: `network_vrt.py` 第 995-1228 行

**功能**: 组合 TMSAG、特征对齐和 reshape 操作

**结构**:
```python
Stage
├── reshape: 空间分辨率调整
│   ├── 'none': 保持尺寸
│   ├── 'down': 下采样 2x (空间维度减半)
│   └── 'up': 上采样 2x (空间维度加倍)
│
├── residual_group1: TMSAG (mut_attn=True)
│   └── 75% 的层使用互自注意力
│
├── linear1: 投影层
│
├── residual_group2: TMSAG (mut_attn=False)
│   └── 25% 的层仅使用自注意力
│
├── linear2: 投影层
│
└── parallel warping (如果 pa_frames > 0)
    ├── DCN 对齐
    └── MLP 融合
```

**张量形状变化示例 (Stage 2, down)**:
```
输入: (B, 120, 6, 224, 224)
    ↓ reshape (down)
(B, 120, 6, 112, 112)
    ↓ TMSAG + Linear
(B, 120, 6, 112, 112)
    ↓ parallel warping
(B, 120, 6, 112, 112)
```

### 3.7 Stage 8 (RTMSA 层)

**位置**: `network_vrt.py` 第 1343-1366 行, 939-992 行

**功能**: 残差时序互自注意力，用于最终特征精炼

**结构**:
```python
# Stage 8 第一部分: 维度提升
Rearrange('n c d h w -> n d h w c')
LayerNorm(embed_dims[6])
Linear(embed_dims[6], embed_dims[7])  # 120 → 180
Rearrange('n d h w c -> n c d h w')

# Stage 8 第二部分: 6个 RTMSA 层
RTMSA (窗口大小: [1, 8, 8] for indep_reconsts, else [6, 8, 8])
```

**特点**:
- 独立重建层 (`indep_reconsts=[11, 12]`) 使用 `window_size=[1, 8, 8]`
  - 时间窗口为 1，表示独立处理每一帧
- 其他层使用 `window_size=[6, 8, 8]` 保持时序建模

### 3.8 输出层

**位置**: `network_vrt.py` 第 1368-1387 行

**视频去模糊模式** (`upscale=1`):
```python
self.conv_after_body = nn.Linear(embed_dims[-1], embed_dims[0])  # 180 → 120
self.conv_last = nn.Conv3d(embed_dims[0], out_chans, kernel_size=(1,3,3), padding=(0,1,1))  # 120 → 3
```

**张量形状变化**:
```
输入: (B, 180, 6, 224, 224)  # Stage 8 输出
    ↓ conv_after_body
(B, 120, 6, 224, 224)
    ↓ conv_last
(B, 3, 6, 224, 224)  # RGB 输出
    ↓ transpose(1, 2)
(B, 6, 3, 224, 224)
```

---

## 4. 完整数据流

### 4.1 前向传播流程

#### 4.1.1 整体流程

```
┌─────────────────────────────────────────────────────────────────┐
│ 输入: (B, N, C, H, W)                                          │
│   - RGB: C=3, RGB+Spike: C=4                                   │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 1. 光流估计 (SpyNet)                                            │
│    - 输入: RGB 通道 (B, N, 3, H, W)                             │
│    - 输出: flows_backward, flows_forward (各4个尺度)            │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. 特征对齐 (get_aligned_image_2frames)                         │
│    - 使用光流对齐前后帧                                          │
│    - 输入: (B, N, C, H, W)                                      │
│    - 输出: x_backward, x_forward (各 B, N, C, H, W)            │
│    - 拼接: [x, x_backward, x_forward] → (B, N, 3*C, H, W)      │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. 输入层 (conv_first)                                           │
│    - 输入: (B, 3*C, N, H, W) 转置后                             │
│    - 输出: (B, 120, N, H, W)                                    │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. 特征提取主网络 (forward_features)                            │
│                                                                 │
│    Stage 1: (B, 120, 6, 224, 224)                              │
│       ↓ reshape (none)                                          │
│    (B, 120, 6, 224, 224)                                       │
│       ↓ TMSAG + Linear + Parallel Warping                      │
│    x1 = (B, 120, 6, 224, 224)                                   │
│                                                                 │
│    Stage 2: (B, 120, 6, 224, 224)                              │
│       ↓ reshape (down)                                          │
│    (B, 120, 6, 112, 112)                                       │
│       ↓ TMSAG + Linear + Parallel Warping                      │
│    x2 = (B, 120, 6, 112, 112)                                   │
│                                                                 │
│    Stage 3: (B, 120, 6, 112, 112)                              │
│       ↓ reshape (down)                                          │
│    (B, 120, 6, 56, 56)                                          │
│       ↓ TMSAG + Linear + Parallel Warping                      │
│    x3 = (B, 120, 6, 56, 56)                                     │
│                                                                 │
│    Stage 4: (B, 120, 6, 56, 56)                                 │
│       ↓ reshape (down)                                          │
│    (B, 120, 6, 28, 28)                                          │
│       ↓ TMSAG + Linear + Parallel Warping                      │
│    x4 = (B, 120, 6, 28, 28)                                     │
│                                                                 │
│    Stage 5: (B, 120, 6, 28, 28)                                 │
│       ↓ reshape (up)                                            │
│    (B, 120, 6, 56, 56)                                          │
│       ↓ TMSAG + Linear + Parallel Warping                      │
│    x = (B, 120, 6, 56, 56)                                      │
│                                                                 │
│    Stage 6: x + x3 = (B, 120, 6, 56, 56)                       │
│       ↓ reshape (up)                                            │
│    (B, 120, 6, 112, 112)                                       │
│       ↓ TMSAG + Linear + Parallel Warping                      │
│    x = (B, 120, 6, 112, 112)                                    │
│                                                                 │
│    Stage 7: x + x2 = (B, 120, 6, 112, 112)                      │
│       ↓ reshape (up)                                            │
│    (B, 120, 6, 224, 224)                                        │
│       ↓ TMSAG + Linear + Parallel Warping                      │
│    x = (B, 120, 6, 224, 224)                                    │
│                                                                 │
│    Stage 8: x + x1 = (B, 120, 6, 224, 224)                      │
│       ↓ 维度提升: 120 → 180                                     │
│    (B, 180, 6, 224, 224)                                        │
│       ↓ 6个 RTMSA 层                                           │
│    (B, 180, 6, 224, 224)                                        │
│       ↓ LayerNorm                                              │
│    (B, 180, 6, 224, 224)                                        │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. 输出层                                                        │
│    - conv_after_body: 180 → 120                                 │
│    - conv_last: 120 → 3 (RGB)                                   │
│    - 输出: (B, N, 3, H, W)                                       │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. 残差连接                                                      │
│    output = x + x_lq[:, :, :3, :, :]                           │
│    - 输出: (B, N, 3, H, W)                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 关键张量形状变化表

| 位置 | 输入形状 | 操作 | 输出形状 | 说明 |
|------|---------|------|---------|------|
| **输入** | `(B, N, C, H, W)` | - | - | C=3(RGB) 或 4(RGB+Spike) |
| **光流估计** | `(B, N, 3, H, W)` | SpyNet | `(B, N-1, 2, H, W)` ×4 | 4个尺度 |
| **特征对齐** | `(B, N, C, H, W)` | flow_warp | `(B, N, C, H, W)` ×2 | backward + forward |
| **对齐拼接** | `(B, N, C, H, W)` ×3 | concat | `(B, N, 3*C, H, W)` | |
| **conv_first** | `(B, 3*C, N, H, W)` | Conv3d | `(B, 120, N, H, W)` | embed_dims[0]=120 |
| **Stage 1** | `(B, 120, 6, 224, 224)` | TMSAG+Warp | `(B, 120, 6, 224, 224)` | x1 (保存) |
| **Stage 2** | `(B, 120, 6, 224, 224)` | Down+TMSAG | `(B, 120, 6, 112, 112)` | x2 (保存) |
| **Stage 3** | `(B, 120, 6, 112, 112)` | Down+TMSAG | `(B, 120, 6, 56, 56)` | x3 (保存) |
| **Stage 4** | `(B, 120, 6, 56, 56)` | Down+TMSAG | `(B, 120, 6, 28, 28)` | x4 |
| **Stage 5** | `(B, 120, 6, 28, 28)` | Up+TMSAG | `(B, 120, 6, 56, 56)` | |
| **Stage 6** | `(B, 120, 6, 56, 56)` | Up+TMSAG | `(B, 120, 6, 112, 112)` | +x3 |
| **Stage 7** | `(B, 120, 6, 112, 112)` | Up+TMSAG | `(B, 120, 6, 224, 224)` | +x2 |
| **Stage 8 输入** | `(B, 120, 6, 224, 224)` | +x1 | `(B, 120, 6, 224, 224)` | |
| **Stage 8 提升** | `(B, 120, 6, 224, 224)` | Linear | `(B, 180, 6, 224, 224)` | embed_dims[7]=180 |
| **Stage 8 RTMSA** | `(B, 180, 6, 224, 224)` | 6×RTMSA | `(B, 180, 6, 224, 224)` | |
| **conv_after_body** | `(B, 180, 6, 224, 224)` | Linear | `(B, 120, 6, 224, 224)` | |
| **conv_last** | `(B, 120, 6, 224, 224)` | Conv3d | `(B, 3, 6, 224, 224)` | RGB 输出 |
| **转置** | `(B, 3, 6, 224, 224)` | transpose(1,2) | `(B, 6, 3, 224, 224)` | |
| **残差连接** | `(B, 6, 3, 224, 224)` | +x_lq | `(B, 6, 3, 224, 224)` | 最终输出 |

### 4.3 RGB+Spike 数据流特殊处理

#### 4.3.1 输入处理

```
RGB 输入: (B, N, 3, H, W)
Spike 输入: (B, N, 1, H, W)
    ↓ concat (在数据加载阶段)
RGB+Spike: (B, N, 4, H, W)
```

#### 4.3.2 光流计算

```python
# 仅使用 RGB 通道 (前3个通道)
if c > 3:
    x_rgb = x[:, :, :3, :, :]  # (B, N, 3, H, W)
flows = self.spynet(x_rgb)  # SpyNet 仅处理 RGB
```

#### 4.3.3 特征对齐

```python
# 对齐操作使用完整的 4 通道 (RGB+Spike)
x_backward, x_forward = get_aligned_image_2frames(x)  # (B, N, 4, H, W)

# 拼接
x = torch.cat([x, x_backward, x_forward], 2)  # (B, N, 12, H, W)
```

#### 4.3.4 conv_first 输入

```python
# RGB+Spike: 4通道, pa_frames=2
conv_first_in_chans = 4 * (1 + 2*2) = 36

# 输入: (B, N, 12, H, W) → (B, 36, N, H, W)
# 输出: (B, 120, N, H, W)
```

#### 4.3.5 输出处理

```python
# 输出为 3 通道 (RGB)
x = self.conv_last(x)  # (B, N, 3, H, W)

# 残差连接时，只使用 RGB 通道
if x_lq.shape[2] != x.shape[2]:  # 4 != 3
    return x + x_lq[:, :, :3, :, :]  # 只用前3个通道
```

---

## 5. 前向传播详细流程

### 5.1 代码调用链

```
VRT.forward()
    ├── get_flows() → SpyNet 光流估计
    ├── get_aligned_image_2frames() → 特征对齐
    ├── conv_first() → 输入投影
    ├── forward_features() → 特征提取
    │   ├── Stage 1-7 → 多尺度特征提取
    │   └── Stage 8 → 最终精炼
    ├── conv_after_body() → 维度降低
    ├── conv_last() → RGB 输出
    └── 残差连接
```

### 5.2 forward_features 详细流程

**位置**: `network_vrt.py` 第 1597-1616 行

```python
def forward_features(self, x, flows_backward, flows_forward):
    # Stage 1-4: 下采样路径
    x1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4])  # 使用第0个尺度光流
    x2 = self.stage2(x1, flows_backward[1::4], flows_forward[1::4])   # 使用第1个尺度光流
    x3 = self.stage3(x2, flows_backward[2::4], flows_forward[2::4])   # 使用第2个尺度光流
    x4 = self.stage4(x3, flows_backward[3::4], flows_forward[3::4])   # 使用第3个尺度光流
    
    # Stage 5-7: 上采样路径 (带残差连接)
    x = self.stage5(x4, flows_backward[2::4], flows_forward[2::4])    # 使用第2个尺度光流
    x = self.stage6(x + x3, flows_backward[1::4], flows_forward[1::4])  # 残差连接 x3
    x = self.stage7(x + x2, flows_backward[0::4], flows_forward[0::4])  # 残差连接 x2
    x = x + x1  # 残差连接 x1
    
    # Stage 8: RTMSA 精炼
    for layer in self.stage8:
        x = layer(x)
    
    # LayerNorm
    x = rearrange(x, 'n c d h w -> n d h w c')
    x = self.norm(x)
    x = rearrange(x, 'n d h w c -> n c d h w')
    
    return x
```

**光流使用说明**:
- `flows_backward[0::4]`: 每4个取第1个 → 使用第0尺度 (原始分辨率)
- `flows_backward[1::4]`: 每4个取第2个 → 使用第1尺度 (1/2分辨率)
- `flows_backward[2::4]`: 每4个取第3个 → 使用第2尺度 (1/4分辨率)
- `flows_backward[3::4]`: 每4个取第4个 → 使用第3尺度 (1/8分辨率)

### 5.3 Stage 前向传播

**位置**: `network_vrt.py` 第 1095-1105 行

```python
def forward(self, x, flows_backward, flows_forward):
    # 1. Reshape (下采样/上采样/不变)
    x = self.reshape(x)  # (B, C, D, H, W)
    
    # 2. 第一组: 互自注意力 (75% 的层)
    x = self.linear1(
        self.residual_group1(x).transpose(1, 4)
    ).transpose(1, 4) + x
    
    # 3. 第二组: 仅自注意力 (25% 的层)
    x = self.linear2(
        self.residual_group2(x).transpose(1, 4)
    ).transpose(1, 4) + x
    
    # 4. Parallel Warping (如果 pa_frames > 0)
    if self.pa_frames:
        x = x.transpose(1, 2)  # (B, D, C, H, W)
        x_backward, x_forward = get_aligned_feature_2frames(x, flows_backward, flows_forward)
        x = self.pa_fuse(
            torch.cat([x, x_backward, x_forward], 2).permute(0, 1, 3, 4, 2)
        ).permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
    
    return x
```

### 5.4 TMSA 前向传播

**位置**: `network_vrt.py` 第 832-852 行

```python
def forward(self, x, mask_matrix):
    # 1. 自注意力 + 互注意力
    if self.use_checkpoint_attn:
        x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
    else:
        x = x + self.forward_part1(x, mask_matrix)  # 残差连接
    
    # 2. MLP
    if self.use_checkpoint_ffn:
        x = x + checkpoint.checkpoint(self.forward_part2, x)
    else:
        x = x + self.forward_part2(x)  # 残差连接
    
    return x
```

**forward_part1** (注意力):
```python
def forward_part1(self, x, mask_matrix):
    # LayerNorm
    x = self.norm1(x)
    
    # 窗口填充
    x = F.pad(x, ...)  # 填充到窗口大小的倍数
    
    # 循环移位 (如果有 shift)
    if shift_size > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
    
    # 窗口分割
    x_windows = window_partition(x, window_size)  # (B*nW, Wd*Wh*Ww, C)
    
    # WindowAttention
    attn_windows = self.attn(x_windows, mask=mask_matrix)
    
    # 窗口合并
    x = window_reverse(attn_windows, window_size, ...)
    
    # 反向循环移位
    if shift_size > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
    
    # DropPath
    x = self.drop_path(x)
    
    return x
```

**forward_part2** (MLP):
```python
def forward_part2(self, x):
    return self.drop_path(self.mlp(self.norm2(x)))
```

---

## 6. 后向传播 (训练流程)

### 6.1 训练循环

**位置**: `model_plain.py` 第 163-177 行

```python
def optimize_parameters(self, current_step):
    # 1. 梯度清零
    self.G_optimizer.zero_grad()
    
    # 2. 前向传播
    self.netG_forward()  # self.E = self.netG(self.L)
    
    # 3. 计算损失
    G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
    
    # 4. 反向传播
    G_loss.backward()
    
    # 5. 梯度裁剪 (可选)
    if G_optimizer_clipgrad > 0:
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), 
            max_norm=self.opt_train['G_optimizer_clipgrad']
        )
    
    # 6. 参数更新
    self.G_optimizer.step()
```

### 6.2 损失函数

**常用损失函数** (配置中可设置):

1. **L1 Loss**: `nn.L1Loss()`
   ```python
   loss = |pred - target|
   ```

2. **L2 Loss (MSE)**: `nn.MSELoss()`
   ```python
   loss = (pred - target)^2
   ```

3. **Charbonnier Loss**: `CharbonnierLoss(eps=1e-6)`
   ```python
   loss = sqrt((pred - target)^2 + eps^2)
   ```
   - 平滑版的 L1，避免 L2 的梯度爆炸

4. **SSIM Loss**: `SSIMLoss()`
   ```python
   loss = 1 - SSIM(pred, target)
   ```
   - 结构相似性损失

### 6.3 梯度流动路径

```
Loss (标量)
    ↓ backward()
conv_last
    ↓
conv_after_body
    ↓
Stage 8 (RTMSA)
    ├─→ Stage 7 (残差分支)
    ├─→ Stage 6 (残差分支)
    └─→ Stage 5 (残差分支)
        ↓
    Stage 4
        ↓
    Stage 3
        ↓
    Stage 2
        ↓
    Stage 1
        ↓
conv_first
    ↓
输入
```

**残差连接的作用**:
- 提供梯度的高速通道
- 避免梯度消失
- 加速训练收敛

### 6.4 优化器配置

**位置**: `model_plain.py` 第 107-119 行

```python
# Adam 优化器
self.G_optimizer = Adam(
    G_optim_params, 
    lr=self.opt_train['G_optimizer_lr'],        # 学习率: 4e-4
    betas=self.opt_train['G_optimizer_betas'],  # (0.9, 0.99)
    weight_decay=self.opt_train['G_optimizer_wd']  # 权重衰减
)
```

**学习率调度**:
- **MultiStepLR**: 在指定 milestones 处降低学习率
- **CosineAnnealingWarmRestarts**: 余弦退火重启

### 6.5 VRT 特殊训练策略

**位置**: `model_vrt.py` 第 65-77 行

#### 6.5.1 固定部分参数训练

```python
# 前 fix_iter 步固定光流网络参数
if current_step < self.fix_iter:
    for name, param in self.netG.named_parameters():
        if any([key in name for key in self.fix_keys]):
            param.requires_grad_(False)  # 固定 SpyNet 参数
```

**原理**: SpyNet 是预训练模型，初期固定其参数可以稳定训练。

#### 6.5.2 分层学习率

```python
# 光流网络使用较小的学习率
flow_params = [param for name, param in ... if 'spynet' in name]
normal_params = [param for name, param in ... if 'spynet' not in name]

G_optim_params = [
    {'params': normal_params, 'lr': base_lr},
    {'params': flow_params, 'lr': base_lr * fix_lr_mul}  # 例如 0.1
]
```

---

## 7. 关键参数说明

### 7.1 模型架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `upscale` | 1 | 上采样因子 (去模糊=1, 超分辨率=4) |
| `in_chans` | 3 | 输入通道数 (RGB=3, RGB+Spike=4) |
| `out_chans` | 3 | 输出通道数 (RGB=3) |
| `img_size` | `[6, 64, 64]` | 输入尺寸 [时间, 高度, 宽度] |
| `window_size` | `[6, 8, 8]` | 注意力窗口大小 |
| `depths` | `[8,8,8,8,8,8,8,4,4,4,4,4,4]` | 各阶段深度 |
| `embed_dims` | `[120,120,...,180,180,...]` | 各阶段特征维度 |
| `num_heads` | `[6,6,...]` | 各阶段注意力头数 |
| `pa_frames` | 2 | 并行对齐帧数 (前后各2帧) |
| `deformable_groups` | 16 | 可变形卷积组数 |

### 7.2 训练超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `batch_size` | 3 | 批次大小 |
| `G_optimizer_lr` | 4e-4 | 初始学习率 |
| `G_optimizer_betas` | `(0.9, 0.99)` | Adam 动量参数 |
| `G_optimizer_wd` | 0 | 权重衰减 |
| `G_lossfn_type` | `charbonnier` | 损失函数类型 |
| `total_iter` | 300000 | 总训练步数 |
| `G_scheduler_type` | `MultiStepLR` | 学习率调度器 |

---

## 8. 模型复杂度分析

### 8.1 参数量估算

假设输入尺寸 `(B, 6, 4, 224, 224)`:

1. **conv_first**: 
   - `36 * 120 * 1 * 3 * 3 = 38,880` 参数

2. **Stage 1-7** (每阶段):
   - TMSAG: `O(C^2 * depth)` → 约 `120^2 * 8 = 115,200` (简化)
   - Linear: `120 * 120 = 14,400`
   - Parallel Warping: `DCN + MLP` → 约 `50,000`
   
3. **Stage 8**:
   - 维度提升: `120 * 180 = 21,600`
   - RTMSA: `180^2 * 4 = 129,600` (每层)
   - 共 6 层: `778,200`

4. **输出层**:
   - `conv_after_body`: `180 * 120 = 21,600`
   - `conv_last`: `120 * 3 * 1 * 3 * 3 = 3,240`

**总参数量**: 约 **20-30M** (根据配置)

### 8.2 计算量 (FLOPs)

假设输入 `(1, 6, 4, 224, 224)`:

1. **卷积操作**: `O(C_in * C_out * K * H * W * D)`
2. **注意力操作**: `O(N^2 * C)`，其中 `N = window_size[0]*window_size[1]*window_size[2]`
   - 单窗口: `6*8*8 = 384` tokens
   - 注意力: `384^2 * 120 = 17,694,720` FLOPs (每窗口)

**总计算量**: 约 **100-200 GFLOPs** (取决于实现)

---

## 9. 注意事项与最佳实践

### 9.1 内存优化

1. **梯度检查点 (Checkpointing)**:
   ```python
   use_checkpoint_attn=True  # 注意力使用 checkpoint
   use_checkpoint_ffn=True    # MLP 使用 checkpoint
   ```
   - 以时间换内存，减少约 40-50% 显存占用

2. **窗口大小调整**:
   - 较小窗口 `[4, 4, 4]` 可降低内存需求
   - 但可能影响性能

3. **批次大小**:
   - 视频模型内存占用大，通常 `batch_size=1-4`

### 9.2 RGB+Spike 输入注意事项

1. **光流计算**: 仅使用 RGB 通道
   - 保证 SpyNet 预训练权重有效性
   - Spike 通道不参与光流估计

2. **特征对齐**: 使用完整 4 通道
   - Spike 信息参与特征提取
   - 对齐后的特征融合包含多模态信息

3. **残差连接**: 仅使用 RGB 通道
   - 输出为 3 通道 RGB
   - 残差连接只使用前 3 个通道

### 9.3 训练技巧

1. **渐进式训练**:
   - 先固定 SpyNet，训练其他部分
   - 再微调全部参数

2. **学习率调度**:
   - 使用 MultiStepLR 在关键点降低学习率
   - 或使用余弦退火

3. **数据增强**:
   - 随机裁剪、翻转、旋转
   - RGB 和 Spike 同步增强

---

## 10. 总结

VRT 是一个高效的视频恢复 Transformer 模型，主要特点包括:

1. **多尺度特征提取**: U型结构提取不同尺度的特征
2. **时序建模**: 窗口自注意力和互注意力建模帧间关系
3. **精确对齐**: 光流+可变形卷积实现亚像素级对齐
4. **残差学习**: 多层次残差连接保证训练稳定

在 RGB+Spike 场景下，模型通过智能的通道分离策略:
- 光流使用 RGB (预训练兼容)
- 特征提取使用完整多模态信息
- 输出保持 RGB 格式

这保证了模型既利用了 Spike 的高时间分辨率优势，又保持了预训练权重的有效性。

---

## 参考文献

1. [VRT: A Video Restoration Transformer](https://arxiv.org/abs/2201.00000)
2. [SpyNet: Learning Optical Flow in 30ms](https://arxiv.org/abs/1611.00850)
3. [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)

---

**文档版本**: v1.0  
**最后更新**: 2025-01-XX  
**维护者**: KAIR Team

