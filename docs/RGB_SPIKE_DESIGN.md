# RGB+Spike 融合关键设计决策

## 完整数据流水线总结

本节提供 RGB+Spike 融合的端到端数据流水线概览，展示从数据加载到模型输出的完整过程，详细追踪 RGB 与 Spike 通道在 pipeline 中的每一步变化。

### 1. 数据加载阶段（真实concat）

**文件**: `data/dataset_video_train_rgbspike.py`

#### 1.1 RGB 数据加载
**位置**: `_load_raw_frame` 方法 (第 385-432 行)

```python
# 从文件系统或 LMDB 加载 RGB 图像
img_bytes = fc.get(img_lq_path, 'lq')
img_lq = utils_video.imfrombytes(img_bytes, float32=True)
img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)  # 转换为 RGB
# 输出: (H, W, 3), dtype=float32, 值域 [0, 1]
```

#### 1.2 Spike 数据加载与体素化
**位置**: `_load_raw_frame` 方法 (第 407-425 行)

```python
# 加载 Spike .dat 文件
spike_file = self.spike_root / clip_name / 'spike' / f'{neighbor:{self.filename_tmpl}}.dat'
spike_stream = SpikeStreamSimple(str(spike_file), spike_h=250, spike_w=400)
spike_matrix = spike_stream.get_spike_matrix(flipud=True)  # (T, H, W)

# TFP 体素化：将时间序列转换为多通道体素
spike_voxel = voxelize_spikes_tfp(
    spike_matrix,
    num_channels=self.spike_channels,  # 默认 4 通道
    device=self._select_tfp_device(),
    half_win_length=self.tfp_half_win_length,  # 默认 20
)  # 输出: (S, H, W), 其中 S=4 (TFP 体素通道数)
```

**关键参数**:
- `spike_channels`: 默认 4（TFP 体素化通道数）
- `spike_h`, `spike_w`: 默认 250×400（Spike 相机原始分辨率）

#### 1.3 分辨率对齐
**位置**: `__getitem__` 方法 (第 223-237 行)

```python
# RGB: 随机裁剪到训练尺寸
img_gts, img_lqs = utils_video.paired_random_crop(
    img_gts, img_lqs, self.gt_size, self.scale, img_gt_path_reference
)
# img_lqs[0].shape: (cropped_h, cropped_w, 3)

# Spike: 调整到与 RGB 相同的空间尺寸
cropped_h, cropped_w = img_lqs[0].shape[:2]
spike_voxels_resized = []
for spike_voxel in spike_voxels:
    spike_voxel_resized = []
    for ch in range(self.spike_channels):  # S=4
        spike_ch = spike_voxel[ch]  # (H, W)
        spike_ch_resized = cv2.resize(
            spike_ch, (cropped_w, cropped_h), 
            interpolation=cv2.INTER_LINEAR
        )
        spike_voxel_resized.append(spike_ch_resized)
    spike_voxel_resized = np.stack(spike_voxel_resized, axis=0)  # (S, H, W)
    spike_voxels_resized.append(spike_voxel_resized)
```

#### 1.4 通道拼接（真实 concat）
**位置**: `__getitem__` 方法 (第 239-251 行)

```python
# 通道顺序：
#   0~2: RGB (0-1 float)
#   3~6: Spike TFP Voxels (0-1 float, 4 channels)
# Total: 7 channels

img_lqs_with_spike = []
for img_lq, spike_voxel in zip(img_lqs, spike_voxels_resized):
    # img_lq: (H, W, 3), spike_voxel: (S, H, W), S=4
    spike_voxel_hwc = np.transpose(spike_voxel, (1, 2, 0))  # (H, W, S)
    # 沿通道维度拼接
    img_lq_with_spike = np.concatenate([img_lq, spike_voxel_hwc], axis=2)
    # 输出: (H, W, 7) = [R, G, B, Spike_0, Spike_1, Spike_2, Spike_3]
    img_lqs_with_spike.append(img_lq_with_spike)
```

#### 1.5 数据增强与归一化
**位置**: `__getitem__` 方法 (第 253-266 行)

```python
# 数据增强（翻转、旋转）
img_lqs_with_spike.extend(img_gts)
img_results = utils_video.augment(
    img_lqs_with_spike, 
    self.opt['use_hflip'], 
    self.opt['use_rot']
)

# 转换为 Tensor: (H, W, C) → (C, H, W)
img_results = utils_video.img2tensor(img_results, bgr2rgb=False)
img_gts = torch.stack(img_results[len(img_lqs_with_spike) // 2:], dim=0)
img_lqs = torch.stack(img_results[:len(img_lqs_with_spike) // 2], dim=0)
# img_lqs: (T, 7, H, W), img_gts: (T, 3, H, W)

# 通道归一化
img_lqs = self._apply_channel_normalization(img_lqs)
# RGB 通道: ImageNet 归一化（如果启用）
# Spike 通道: 可选归一化（如果配置）
```

**归一化配置** (第 107-108 行):
```python
self.rgb_norm_stats = self._build_norm_stats(
    opt.get('rgb_normalize', None), num_channels=3, preset='imagenet'
)
self.spike_norm_stats = self._build_norm_stats(
    opt.get('spike_normalize', None), num_channels=self.spike_channels
)
```

**归一化应用** (第 548-554 行):
```python
def _apply_channel_normalization(self, tensor):
    """Apply RGB ImageNet-style normalization and optional spike scaling."""
    if self.rgb_norm_stats is not None:
        tensor[:, :3, :, :] = (tensor[:, :3, :, :] - self.rgb_norm_stats['mean']) / self.rgb_norm_stats['std']
    if self.spike_norm_stats is not None and tensor.size(1) > 3:
        tensor[:, 3:, :, :] = (tensor[:, 3:, :, :] - self.spike_norm_stats['mean']) / self.spike_norm_stats['std']
    return tensor
```

**最终输出**:
- `LQ`: `(T, 7, H, W)` - 7 通道 = `[R, G, B, Spike_0, Spike_1, Spike_2, Spike_3]`
- `GT`: `(T, 3, H, W)` - 3 通道 = `[R, G, B]`

### 2. 模型初始化与输入层适配

**文件**: `models/network_vrt.py` - `VRT.__init__` 方法

#### 2.1 模型参数配置
**位置**: `models/network_vrt.py` 第 1333-1362 行

```python
def __init__(self,
             upscale=4,
             in_chans=3,      # 输入通道数：RGB=3, RGB+Spike=7
             out_chans=3,     # 输出通道数：RGB=3
             img_size=[6, 64, 64],
             window_size=[6, 8, 8],
             depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
             # ... 其他参数 ...
             pa_frames=2,     # 时间对齐帧数（前后各 pa_frames 帧）
             # ...):
    super().__init__()
    self.in_chans = in_chans      # 保存输入通道数（7 for RGB+Spike）
    self.out_chans = out_chans    # 保存输出通道数（3 for RGB）
    self.pa_frames = pa_frames    # 时间对齐帧数
```

**配置文件示例**: `options/gopro_rgbspike_local.json` 第 278-285 行
```json
"netG": {
    "net_type": "vrt",
    "in_chans": 7,  // 3 (RGB) + 4 (Spike TFP)
    "out_chans": 3,
    "pa_frames": 4,  // 前后各 4 帧对齐
    // ...
}
```

#### 2.2 第一层卷积通道计算
**位置**: `models/network_vrt.py` 第 1372-1382 行

```python
# conv_first 输入通道数计算
if self.pa_frames:
    # 当 pa_frames > 0 时，使用时间对齐
    # 对齐后通道数 = in_chans * 9
    # 9 = 1 (当前帧) + 4 (后向对齐) + 4 (前向对齐)
    # 注意：实际对齐使用 nearest4 模式，每个通道扩展为 4 倍
    if self.nonblind_denoising:
        conv_first_in_chans = in_chans * 9 + 1  # +1 for noise level map
    else:
        conv_first_in_chans = in_chans * 9
else:
    conv_first_in_chans = in_chans

# 创建第一层 3D 卷积
self.conv_first = nn.Conv3d(
    conv_first_in_chans,      # 输入通道：7 * 9 = 63 (RGB+Spike, pa_frames=4)
    embed_dims[0],            # 输出通道：embed_dims[0] (例如 96)
    kernel_size=(1, 3, 3),     # 时间维度不降采样，空间 3x3 卷积
    padding=(0, 1, 1)
)
```

**通道计算详解**:
- **输入通道数**: `in_chans = 7` (RGB 3 + Spike 4)
- **时间对齐扩展**: `pa_frames = 4` 时，使用 `nearest4` 模式对齐
  - 当前帧: `7` 通道
  - 后向对齐: `7 × 4 = 28` 通道（nearest4 模式）
  - 前向对齐: `7 × 4 = 28` 通道（nearest4 模式）
  - **总计**: `7 + 28 + 28 = 63` 通道
- **注意**: 虽然代码中写的是 `in_chans * 9`，但实际对齐使用 `nearest4` 模式，每个通道扩展为 4 倍，所以实际是 `in_chans * (1 + 4 + 4) = in_chans * 9`，但每个对齐分支是 4 倍扩展。

**实际通道数验证**:
- 配置: `in_chans=7`, `pa_frames=4`
- `conv_first.in_channels = 63`
- 输入到 `conv_first` 的 tensor 形状: `(B, 63, N, H, W)`

### 3. 前向传播流程

#### 3.1 输入验证
**位置**: `models/network_vrt.py` - `forward` 方法 (第 1510-1521 行)

```python
def forward(self, x):
    # x: (B, N, C, H, W) = (B, N, 7, H, W) for RGB+Spike
    timer = getattr(self, 'timer', None)
    
    if self.pa_frames:
        # 非盲去噪模式：分离噪声水平图
        if self.nonblind_denoising:
            x, noise_level_map = x[:, :, :self.in_chans, :, :], x[:, :, self.in_chans:, :, :]
        
        # 保存原始输入用于残差连接
        x_lq = x.clone()  # (B, N, 7, H, W)
        x_lq_rgb = self.extract_rgb(x_lq)  # (B, N, 3, H, W) - 仅 RGB 通道
```

**extract_rgb 方法** (第 1502-1504 行):
```python
def extract_rgb(self, x, channels=3):
    """Return up to the first `channels` feature channels (default: RGB)."""
    return x[:, :, :min(channels, x.size(2)), :, :]
```

#### 3.2 光流计算（智能分离 RGB 通道）
**位置**: `models/network_vrt.py` - `get_flow_2frames` 方法 (第 1634-1655 行)

```python
def get_flow_2frames(self, x):
    '''Get flow between frames t and t+1 from x.'''
    
    b, n, c, h, w = x.size()  # c = 7 (RGB+Spike)
    
    # SpyNet 在 RGB 图像上预训练，只能接受 3 通道
    # 使用 extract_rgb 提取前 3 个 RGB 通道
    x_flow = self.extract_rgb(x)  # (B, N, 3, H, W)
    c_flow = x_flow.size(2)       # c_flow = 3
    
    # 准备相邻帧对
    x_1 = x_flow[:, :-1, :, :, :].reshape(-1, c_flow, h, w)  # (B*(N-1), 3, H, W)
    x_2 = x_flow[:, 1:, :, :, :].reshape(-1, c_flow, h, w)   # (B*(N-1), 3, H, W)
    
    # 后向光流：从 t+1 到 t
    flows_backward = self.spynet(x_1, x_2)
    flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) 
                      for flow, i in zip(flows_backward, range(4))]
    
    # 前向光流：从 t 到 t+1
    flows_forward = self.spynet(x_2, x_1)
    flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) 
                     for flow, i in zip(flows_forward, range(4))]
    
    return flows_backward, flows_forward
```

**关键点**:
- ✅ **只使用 RGB 通道**: `extract_rgb(x)` 提取前 3 个通道
- ✅ **SpyNet 限制**: SpyNet 在 RGB 上预训练，只能接受 3 通道输入
- ✅ **完整数据保留**: 原始的 7 通道数据 `x` 仍然保留，用于后续特征对齐
- ✅ **光流输出**: 多尺度光流 `[flow0, flow1, flow2, flow3]`，分辨率递减

**SpyNet 初始化** (第 1386 行):
```python
if self.pa_frames:
    self.spynet = SpyNet(spynet_path, [2, 3, 4, 5])  # 返回 4 个尺度的光流
```

#### 3.3 特征对齐（使用完整通道 + 光流）
**位置**: `models/network_vrt.py` - `get_aligned_image_2frames` 方法 (第 1709-1736 行)

```python
def get_aligned_image_2frames(self, x, flows_backward, flows_forward):
    '''Parallel feature warping for 2 frames using nearest4 mode.'''
    
    # x: (B, N, 7, H, W) - 完整的 RGB+Spike 通道
    n = x.size(1)
    
    # 后向对齐：将后续帧对齐到当前帧
    x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
    for i in range(n - 1, 0, -1):
        x_i = x[:, i, ...]  # (B, 7, H, W) - 包含所有通道
        flow = flows_backward[:, i - 1, ...]  # (B, 2, H, W)
        # nearest4 模式：每个通道扩展为 4 倍
        x_backward.insert(0, flow_warp(
            x_i, 
            flow.permute(0, 2, 3, 1),  # (B, H, W, 2)
            'nearest4'  # 关键：每个通道扩展为 4 倍
        ))  # 输出: (B, 28, H, W) = 7 * 4
    
    # 前向对齐：将前序帧对齐到当前帧
    x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
    for i in range(0, n - 1):
        x_i = x[:, i, ...]  # (B, 7, H, W)
        flow = flows_forward[:, i, ...]  # (B, 2, H, W)
        x_forward.append(flow_warp(
            x_i, 
            flow.permute(0, 2, 3, 1), 
            'nearest4'
        ))  # 输出: (B, 28, H, W)
    
    x_backward = torch.stack(x_backward, 1)  # (B, N, 28, H, W)
    x_forward = torch.stack(x_forward, 1)   # (B, N, 28, H, W)
    
    # 通道数验证
    expected_channels = self.in_chans * 4  # 7 * 4 = 28
    if x_backward.size(2) != expected_channels or x_forward.size(2) != expected_channels:
        raise ValueError(
            f"SGP alignment produced mismatched channels "
            f"(expected 4×{self.in_chans}={expected_channels}). "
            f"x_backward: {x_backward.shape}, x_forward: {x_forward.shape}."
        )
    
    return [x_backward, x_forward]
```

**nearest4 模式说明** (第 241-252 行):
```python
def flow_warp(x, flow, interp_mode='bilinear', ...):
    # ...
    if interp_mode == 'nearest4':
        # 对每个通道，使用 4 个最近邻采样点
        # 输出通道数 = 输入通道数 × 4
        output00 = F.grid_sample(x, ..., mode='nearest', ...)
        output01 = F.grid_sample(x, ..., mode='nearest', ...)
        output10 = F.grid_sample(x, ..., mode='nearest', ...)
        output11 = F.grid_sample(x, ..., mode='nearest', ...)
        return torch.cat([output00, output01, output10, output11], 1)
```

**通道扩展过程**:
- 输入: `(B, 7, H, W)` - RGB+Spike 7 通道
- nearest4 对齐: 每个通道扩展为 4 个采样点
- 输出: `(B, 28, H, W)` = `7 × 4` 通道

#### 3.4 通道拼接与 conv_first 输入
**位置**: `models/network_vrt.py` - `forward` 方法 (第 1530-1552 行)

```python
# 特征对齐
x_backward, x_forward = self.get_aligned_image_2frames(
    x, flows_backward[0], flows_forward[0]
)
# x_backward: (B, N, 28, H, W)
# x_forward: (B, N, 28, H, W)

# 拼接：当前帧 + 后向对齐 + 前向对齐
x = torch.cat([x, x_backward, x_forward], 2)
# x: (B, N, 63, H, W) = 7 (当前) + 28 (后向) + 28 (前向)

# 非盲去噪：拼接噪声水平图
if self.nonblind_denoising:
    x = torch.cat([x, noise_level_map], 2)

# 通道数验证
if x.size(2) != self.conv_first.in_channels:
    raise ValueError(
        f"Channel Mismatch Error! \n"
        f"Current x shape: {x.shape} (Channels: {x.size(2)}) \n"
        f"Expected conv_first in_channels: {self.conv_first.in_channels} \n"
        f"Configured in_chans: {self.in_chans} \n"
        f"Mode: {'Train' if self.training else 'Test/Val'} \n"
        f"Hint: Check if your input tensor includes all expected channels "
        f"(e.g. RGB+Spike=7) and that SGP alignment produced the expected 9x expansion."
    )
```

#### 3.5 特征提取（Transformer 主干）
**位置**: `models/network_vrt.py` - `forward_features` 方法 (第 1738-1785 行)

```python
def forward_features(self, x, flows_backward, flows_forward):
    '''Main network for feature extraction.'''
    
    # x: (B, C, N, H, W) - 经过 conv_first 后的特征
    # C = embed_dims[0] (例如 96)
    
    # Stage 1-7: 多尺度特征提取（下采样 + 上采样）
    x1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4])
    x2 = self.stage2(x1, flows_backward[1::4], flows_forward[1::4])
    x3 = self.stage3(x2, flows_backward[2::4], flows_forward[2::4])
    x4 = self.stage4(x3, flows_backward[3::4], flows_forward[3::4])
    x = self.stage5(x4, flows_backward[2::4], flows_forward[2::4])
    x = self.stage6(x + x3, flows_backward[1::4], flows_forward[1::4])
    x = self.stage7(x + x2, flows_backward[0::4], flows_forward[0::4])
    x = x + x1  # 残差连接
    
    # Stage 8: 独立重建层
    for layer in self.stage8:
        x = layer(x)
    
    # 归一化
    x = rearrange(x, 'n c d h w -> n d h w c')
    x = self.norm(x)
    x = rearrange(x, 'n d h w c -> n c d h w')
    
    return x
```

**Stage 结构**: 每个 Stage 使用 DCN (可变形卷积) 进行特征对齐，使用 Transformer 进行特征提取。所有阶段都使用完整的特征通道（不再区分 RGB 和 Spike）。

#### 3.6 输出重建与残差连接
**位置**: `models/network_vrt.py` - `forward` 方法 (第 1554-1570 行)

```python
if self.upscale == 1:
    # 视频去模糊等任务（不进行上采样）
    # 转置: (B, N, 63, H, W) → (B, 63, N, H, W)
    x = self.conv_first(x.transpose(1, 2))
    # x: (B, embed_dims[0], N, H, W)
    
    # 特征提取
    x_features = self.forward_features(x, flows_backward, flows_forward)
    # x_features: (B, embed_dims[-1], N, H, W)
    
    # 残差连接 + 投影
    x = x + self.conv_after_body(x_features.transpose(1, 4)).transpose(1, 4)
    # x: (B, embed_dims[0], N, H, W)
    
    # 输出重建
    x = self.conv_last(x).transpose(1, 2)
    # x: (B, N, 3, H, W) - 输出 RGB 3 通道
    
    # 残差连接：只使用输入的前 3 个 RGB 通道
    return x + x_lq_rgb  # x_lq_rgb: (B, N, 3, H, W)
```

**conv_last 定义** (第 1458-1460 行):
```python
if self.upscale == 1:
    self.conv_last = nn.Conv3d(
        embed_dims[0],      # 输入: embed_dims[0] (例如 96)
        out_chans,          # 输出: 3 (RGB)
        kernel_size=(1, 3, 3), 
        padding=(0, 1, 1)
    )
```

**关键点**:
- ✅ **输出通道**: `out_chans=3` (RGB)
- ✅ **残差连接**: 使用 `x_lq_rgb`（仅 RGB 通道），而不是完整的 7 通道输入
- ✅ **通道匹配**: 输出 3 通道与 GT 的 3 通道匹配

### 4. 完整流水线图与通道追踪

```
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 1: 数据加载 (dataset_video_train_rgbspike.py)                 │
├─────────────────────────────────────────────────────────────────────┤
│ RGB:    从文件/LMDB 加载 → (H, W, 3), float32, [0, 1]              │
│ Spike:  从 .dat 加载 → TFP 体素化 → (4, H_spike, W_spike)          │
│         默认: spike_channels=4, spike_h=250, spike_w=400           │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 2: 分辨率对齐 (__getitem__, 第 223-237 行)                    │
├─────────────────────────────────────────────────────────────────────┤
│ RGB:    paired_random_crop → (cropped_h, cropped_w, 3)             │
│ Spike:  cv2.resize (INTER_LINEAR) → (cropped_h, cropped_w, 4)     │
│         确保 RGB 和 Spike 空间尺寸一致                              │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 3: 通道拼接 ✅ [真实 concat] (__getitem__, 第 239-251 行)      │
├─────────────────────────────────────────────────────────────────────┤
│ np.concatenate([img_lq, spike_voxel_hwc], axis=2)                  │
│ 输出: (H, W, 7) = [R, G, B, Spike_0, Spike_1, Spike_2, Spike_3]   │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 4: 数据增强与归一化 (__getitem__, 第 253-266 行)               │
├─────────────────────────────────────────────────────────────────────┤
│ 增强:   flip, rotate                                                │
│ 转Tensor: (H, W, 7) → (7, H, W)                                    │
│ 归一化:  RGB ImageNet 归一化（可选），Spike 归一化（可选）          │
│ 输出:   LQ (T, 7, H, W), GT (T, 3, H, W)                          │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 5: 模型前向传播 (network_vrt.py forward)                      │
├─────────────────────────────────────────────────────────────────────┤
│ 输入:   (B, N, 7, H, W)                                            │
│         ├─→ extract_rgb → (B, N, 3, H, W) ─→ SpyNet ─→ 光流        │
│         └─→ 完整 (B, N, 7, H, W) ─→ get_aligned_image_2frames      │
│             使用 nearest4 模式: 7 → 28 通道                        │
│ 拼接:   [当前 7] + [后向 28] + [前向 28] = 63 通道                  │
│ conv_first: (B, 63, N, H, W) → (B, embed_dims[0], N, H, W)        │
│ Transformer: 多尺度特征提取（Stage 1-8）                            │
│ conv_last: (B, embed_dims[0], N, H, W) → (B, 3, N, H, W)          │
│ 残差:   输出 + x_lq_rgb (仅 RGB 通道)                               │
│ 输出:   (B, N, 3, H, W) - RGB 3 通道                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 5. 通道维度变化追踪表

| 阶段 | 位置 | 输入形状 | 输出形状 | 通道说明 |
|------|------|---------|---------|---------|
| **数据加载** | `_load_raw_frame` | - | RGB: (H, W, 3)<br>Spike: (4, H_spike, W_spike) | RGB 3 通道，Spike 4 通道 |
| **分辨率对齐** | `__getitem__` | RGB: (H, W, 3)<br>Spike: (4, H_spike, W_spike) | RGB: (H_crop, W_crop, 3)<br>Spike: (4, H_crop, W_crop) | 空间尺寸对齐 |
| **通道拼接** | `__getitem__` | RGB: (H, W, 3)<br>Spike: (4, H, W) | (H, W, 7) | `[R, G, B, S0, S1, S2, S3]` |
| **转 Tensor** | `__getitem__` | (H, W, 7) | (T, 7, H, W) | 批次化 |
| **模型输入** | `forward` | (B, N, 7, H, W) | (B, N, 7, H, W) | 输入验证 |
| **光流计算** | `get_flow_2frames` | (B, N, 7, H, W) | 光流: (B, N-1, 2, H, W) | 仅使用前 3 通道 |
| **特征对齐** | `get_aligned_image_2frames` | (B, N, 7, H, W) | 后向: (B, N, 28, H, W)<br>前向: (B, N, 28, H, W) | nearest4 模式，7×4=28 |
| **通道拼接** | `forward` | 当前: (B, N, 7, H, W)<br>后向: (B, N, 28, H, W)<br>前向: (B, N, 28, H, W) | (B, N, 63, H, W) | 7+28+28=63 |
| **conv_first** | `forward` | (B, N, 63, H, W) | (B, embed_dims[0], N, H, W) | 3D 卷积，时间维度转置 |
| **特征提取** | `forward_features` | (B, embed_dims[0], N, H, W) | (B, embed_dims[-1], N, H, W) | Transformer 主干 |
| **conv_last** | `forward` | (B, embed_dims[0], N, H, W) | (B, 3, N, H, W) | 输出 RGB 3 通道 |
| **残差连接** | `forward` | 输出: (B, N, 3, H, W)<br>输入: (B, N, 3, H, W) | (B, N, 3, H, W) | 仅 RGB 通道残差 |

### 6. 关键优点

1. ✅ **真实拼接**：使用 `np.concatenate` 真实拼接 RGB 和 Spike 通道，不是硬塞
2. ✅ **参数匹配**：`in_chans=7` 正确配置了网络输入（RGB 3 + Spike 4）
3. ✅ **智能分离**：光流计算仅使用 RGB 通道，特征提取使用完整 7 通道
4. ✅ **向后兼容**：纯 RGB 模型（`in_chans=3`）仍然可用
5. ✅ **自动对齐**：Spike 分辨率自动 resize 到 RGB 大小
6. ✅ **通道验证**：每个关键步骤都有通道数验证，确保数据流正确
7. ✅ **可配置归一化**：RGB 和 Spike 通道可独立配置归一化策略

### 设计原则

这是一个**架构级别的优雅设计**，而非简单的数据硬塞：
- **模态特异性**：不同模态在不同阶段发挥作用
- **任务分工**：RGB 负责空间结构，Spike 补充时序信息
- **参数高效**：保持预训练模型权重有效性
- **可解释性**：每个阶段的输入输出都有明确的物理意义

---

## 光流计算只使用 RGB 通道

### 问题
当输入是 RGB+Spike (7通道) 时，SpyNet 光流网络如何处理？

### 解决方案
在 `get_flow_2frames` 方法中，**使用 `extract_rgb` 方法只提取前 3 个 RGB 通道用于光流计算**，Spike 通道不参与光流估计。

### 理由

#### 1. SpyNet 预训练限制
SpyNet 在 RGB 图像上预训练，使用 ImageNet 标准化参数：
```python
self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
```
- 只支持 **3 通道输入**
- 预训练权重对第 4 个通道无效
- 强行输入 4 通道会导致维度不匹配

#### 2. 光流的物理意义
- **光流**：描述像素级别的运动向量 (u, v)
- **RGB 图像**：包含清晰的边缘、纹理、颜色等空间特征，非常适合计算光流
- **Spike 数据**：是事件流的累积，统计特性与 RGB 完全不同

#### 3. Spike 的时间特性
- **光流**：描述的是 **帧间运动** (t → t+1)
- **Spike**：本身就是 **高时间分辨率的运动信息** (微秒级)
- 如果把 Spike 也输入光流网络，会产生 **信息冗余和混淆**

#### 4. 多模态融合分工原则

VRT 的多模态融合采用 **分阶段处理**：

| 阶段 | 模块 | 输入 | 作用 |
|------|------|------|------|
| **1. 光流估计** | SpyNet | **仅 RGB** (3 通道) | 估计帧间运动向量 |
| **2. 特征对齐** | flow_warp (nearest4) | **RGB+Spike** (7 通道) + 光流 | 使用光流引导对齐多模态特征，每个通道扩展为 4 倍 |
| **3. 特征提取** | Transformer | **对齐后的 RGB+Spike** (63 通道) | 深度特征学习和重建 |

这样的设计：
- ✅ **保持 SpyNet 预训练权重有效性**
- ✅ **让 RGB 和 Spike 各司其职**
  - RGB：提供空间结构和运动估计
  - Spike：提供高时间分辨率的运动细节
- ✅ **在对齐阶段才融合多模态信息**

### 代码实现

**文件位置**: `models/network_vrt.py`

**extract_rgb 方法** (第 1502-1504 行):
```python
def extract_rgb(self, x, channels=3):
    """Return up to the first `channels` feature channels (default: RGB)."""
    return x[:, :, :min(channels, x.size(2)), :, :]
```

**get_flow_2frames 方法** (第 1634-1655 行):
```python
def get_flow_2frames(self, x):
    '''Get flow between frames t and t+1 from x.'''

    b, n, c, h, w = x.size()  # c = 7 for RGB+Spike
    
    # SpyNet 在 RGB 图像上预训练，只能接受 3 通道
    # 使用 extract_rgb 提取前 3 个 RGB 通道
    x_flow = self.extract_rgb(x)  # (B, N, 3, H, W)
    c_flow = x_flow.size(2)       # c_flow = 3
    
    # 准备相邻帧对
    x_1 = x_flow[:, :-1, :, :, :].reshape(-1, c_flow, h, w)  # (B*(N-1), 3, H, W)
    x_2 = x_flow[:, 1:, :, :, :].reshape(-1, c_flow, h, w)   # (B*(N-1), 3, H, W)

    # 后向光流：从 t+1 到 t
    flows_backward = self.spynet(x_1, x_2)
    flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) 
                      for flow, i in zip(flows_backward, range(4))]

    # 前向光流：从 t 到 t+1
    flows_forward = self.spynet(x_2, x_1)
    flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) 
                     for flow, i in zip(flows_forward, range(4))]

    return flows_backward, flows_forward
```

### 数据流示意图

```
输入: RGB+Spike [B, N, 7, H, W]
                |
                ├─────────────────┐
                |                 |
                v                 v
        [B,N,3,H,W]         [B,N,7,H,W]
        (extract_rgb)       (完整保留)
        (仅 RGB)            (RGB+Spike)
                |                 |
                v                 |
            SpyNet                |
        (光流估计)                |
        (3 通道输入)              |
                |                 |
                v                 |
        flows [B,N-1,2,H,W]       |
        (多尺度: 4 个尺度)        |
                |                 |
                └────────┬─────────┘
                         v
                   get_aligned_image_2frames
                   (使用完整 7 通道 + 光流)
                   nearest4 模式对齐
                         |
                         v
        [B,N,7,H,W] → [B,N,28,H,W] (后向)
        [B,N,7,H,W] → [B,N,28,H,W] (前向)
                         |
                         v
                   通道拼接
        [当前 7] + [后向 28] + [前向 28] = 63
                         |
                         v
                   conv_first
                   (63 → embed_dims[0])
                         |
                         v
                  Transformer
                  (特征提取, Stage 1-8)
                         |
                         v
                   conv_last
                   (embed_dims[0] → 3)
                         |
                         v
                    输出重建
                  [B, N, 3, H, W]
                  (RGB 3 通道)
```

## 总结

这个设计体现了 **模态特异性** 和 **任务分工** 的原则：
- 光流网络专注于 RGB 的空间运动估计
- Spike 在后续阶段补充时序信息
- 避免了预训练模型的维度冲突
- 保持了模型的可解释性和可扩展性

如果未来需要让 Spike 也参与光流估计，可以：
1. 训练一个 7 通道的 SpyNet（RGB 3 + Spike 4）
2. 或者设计双分支光流网络（RGB 分支 + Spike 分支）
3. 或者使用多模态光流网络（同时接受 RGB 和 Spike 输入）

但当前方案是最简单、最稳定的实现方式，充分利用了 SpyNet 的预训练权重。

---

## 相关文件清单

### 数据加载相关
- **`data/dataset_video_train_rgbspike.py`**: RGB+Spike 数据集实现
  - `__init__`: 数据集初始化，配置 Spike 参数（第 85-185 行）
  - `__getitem__`: 数据加载、对齐、拼接（第 187-271 行）
  - `_load_raw_frame`: 单帧 RGB+Spike 加载（第 385-432 行）
  - `_apply_channel_normalization`: 通道归一化（第 548-554 行）

### 模型相关
- **`models/network_vrt.py`**: VRT 模型实现
  - `VRT.__init__`: 模型初始化，`conv_first` 通道计算（第 1333-1473 行）
  - `VRT.forward`: 前向传播主流程（第 1510-1612 行）
  - `VRT.extract_rgb`: RGB 通道提取（第 1502-1504 行）
  - `VRT.get_flow_2frames`: 光流计算（第 1634-1655 行）
  - `VRT.get_aligned_image_2frames`: 特征对齐（第 1709-1736 行）
  - `VRT.forward_features`: Transformer 特征提取（第 1738-1785 行）

### 配置文件
- **`options/gopro_rgbspike_local.json`**: RGB+Spike 训练配置
  - `netG.in_chans`: 7（RGB 3 + Spike 4）
  - `datasets.train.spike_channels`: 4（TFP 体素化通道数）

### 工具函数
- **`data/spike_recc/spikecv_loader.py`**: Spike 数据加载和 TFP 体素化
  - `voxelize_spikes_tfp`: TFP 体素化函数

### 训练/测试脚本
- **`main_train_vrt.py`**: 训练脚本
- **`main_test_vrt.py`**: 测试脚本，包含通道验证（第 406-418 行）

---

## 配置参数说明

### 数据集配置 (`datasets.train`)
- `spike_channels`: Spike TFP 体素化通道数，默认 4
- `spike_h`, `spike_w`: Spike 相机原始分辨率，默认 250×400
- `rgb_normalize`: RGB 归一化配置（可选，如 'imagenet'）
- `spike_normalize`: Spike 归一化配置（可选，字典格式）

### 模型配置 (`netG`)
- `in_chans`: 输入通道数，RGB+Spike 时为 7（3 + 4）
- `out_chans`: 输出通道数，RGB 为 3
- `pa_frames`: 时间对齐帧数，影响 `conv_first` 输入通道数

### 通道数计算公式
- **数据集输出**: `LQ channels = 3 (RGB) + spike_channels (Spike)`
- **模型输入**: `in_chans = 3 + spike_channels`
- **对齐后通道**: `aligned_channels = in_chans × 4` (nearest4 模式)
- **conv_first 输入**: `conv_first_in_chans = in_chans × 9` (当前 1 + 后向 4 + 前向 4)

