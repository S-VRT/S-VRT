# Paired Spatial Crop：RGB 与 Spike 空间对齐裁剪

> 日期：2026-04-14
> 状态：设计完成，待实现

## 1. 问题

当前训练 data 层的处理流程：

```
RGB (720×1280) → paired_random_crop(gt_size) → patch (160×160)   ← 只覆盖原图一小块
Spike (250×400) → cv2.resize(160×160)                            ← 覆盖整个视场
```

RGB patch 对应原图某个局部区域，但 Spike 是整张图的缩略图。两者空间不对齐。

### 前提

- RGB 和 Spike 的 FOV（视场）完全重合，只是像素密度不同
- 对齐关系是纯比例换算，无需额外标定参数

## 2. 修正后的流程

```
RGB (720×1280) → paired_random_crop → patch (160×160), 返回 crop_params{top, left, patch_size}
                                                         ↓ 按比例换算
Spike (250×400) → crop 对应区域 (~56×50) → cv2.resize → (160×160)
```

### 2.1 比例换算公式

```python
# lq_h, lq_w = RGB LQ 原始空间尺寸（crop 前）
# spike_h, spike_w = Spike 原始空间尺寸
ratio_h = spike_h / lq_h   # e.g. 250/720 ≈ 0.347
ratio_w = spike_w / lq_w   # e.g. 400/1280 = 0.3125

spike_top    = round(top * ratio_h)
spike_left   = round(left * ratio_w)
spike_crop_h = round(patch_size * ratio_h)
spike_crop_w = round(patch_size * ratio_w)

# 边界 clamp（round 可能导致越界）
spike_crop_h = max(spike_crop_h, 1)
spike_crop_w = max(spike_crop_w, 1)
spike_top    = min(spike_top, spike_h - spike_crop_h)
spike_left   = min(spike_left, spike_w - spike_crop_w)
```

### 2.2 各配置下的实际数值

| 配置 | RGB 原始 | gt_size | Spike 原始 | Spike crop 区域 | 上采样倍率 |
|------|---------|---------|-----------|----------------|-----------|
| server | 720×1280 | 160 | 360×640 | 80×80 | 2× |
| local/debug | 720×1280 | 160 | 250×400 | ~56×50 | ~3× |
| 006_train | 720×1280 | 224 | 250×400 | ~78×70 | ~3× |

## 3. 改动范围

### 3.1 `utils/utils_video.py` — `paired_random_crop`

**改动**：返回值增加 crop 参数字典。

现有签名：
```python
def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    ...
    return img_gts, img_lqs
```

改为：
```python
def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    ...
    crop_params = {'top': top, 'left': left, 'lq_patch_size': lq_patch_size}
    return img_gts, img_lqs, crop_params
```

**影响**：所有现有调用方需要适配第三个返回值。非 rgbspike 的 dataset 用 `_` 忽略即可。

### 3.2 `data/dataset_video_train_rgbspike.py` — `__getitem__` (line 292~314)

**改动**：接收 crop 坐标，在 Spike 上按比例 crop 对应区域，再 resize 到 RGB crop 尺寸。

改动前（line 292~314）：
```python
# randomly crop RGB frames
img_gts, img_lqs = utils_video.paired_random_crop(...)

# Resize spike voxels to match the cropped RGB size
cropped_h, cropped_w = img_lqs[0].shape[:2]
for spike_voxel in spike_voxels:
    for ch in range(self.spike_channels):
        spike_ch_resized = cv2.resize(spike_voxel[ch], (cropped_w, cropped_h), ...)
```

改动后：
```python
# randomly crop RGB frames
img_gts, img_lqs, crop_params = utils_video.paired_random_crop(...)

cropped_h, cropped_w = img_lqs[0].shape[:2]
lq_h_orig, lq_w_orig = <crop 前记录的 LQ 原始尺寸>

# 按比例在 Spike 上 crop 对应区域
ratio_h = self.spike_h / lq_h_orig
ratio_w = self.spike_w / lq_w_orig
sp_top    = round(crop_params['top'] * ratio_h)
sp_left   = round(crop_params['left'] * ratio_w)
sp_crop_h = max(round(crop_params['lq_patch_size'] * ratio_h), 1)
sp_crop_w = max(round(crop_params['lq_patch_size'] * ratio_w), 1)
sp_top    = min(sp_top, self.spike_h - sp_crop_h)
sp_left   = min(sp_left, self.spike_w - sp_crop_w)

for spike_voxel in spike_voxels:
    spike_cropped = spike_voxel[:, sp_top:sp_top+sp_crop_h, sp_left:sp_left+sp_crop_w]
    for ch in range(self.spike_channels):
        spike_ch_resized = cv2.resize(spike_cropped[ch], (cropped_w, cropped_h), ...)
```

**注意**：`lq_h_orig, lq_w_orig` 需要在调用 `paired_random_crop` 之前从 `img_lqs[0].shape[:2]` 记录。

### 3.3 `data/dataset_video_train_rgbspike.py` — flow_spike 处理 (line 316~324)

**改动**：`flow_spikes_resized` 也有同样的整张 resize 问题。用相同的 `sp_top, sp_left, sp_crop_h, sp_crop_w` 对 flow_spike 做 paired crop 再 resize。

flow_spike 的原始空间尺寸与 spike_voxel 相同（都来自 Spike 相机），所以复用同一组 crop 参数。

### 3.4 不改的文件

| 文件 | 原因 |
|------|------|
| `data/dataset_video_test.py` | 测试时不做 random crop，spike resize 到 RGB 全分辨率是正确的（FOV 对齐） |
| `models/fusion/adapters/early.py` | 仍然接收同分辨率的 RGB 和 Spike，无需改动 |
| 设计文档中的 SpikeUpsample | 那是为全分辨率推理设计的，与训练 crop 无关 |

## 4. 边界条件

1. **`round()` 越界**：`spike_top + spike_crop_h` 可能超过 `spike_h` → 用 `min()` clamp
2. **极小 crop**：`spike_crop_h/w` 最小为 1，避免空切片
3. **`scale > 1` 的 SR 场景**：`paired_random_crop` 中 `top/left` 是 LQ 坐标系的，换算时用 LQ 原始尺寸（`lq_h_orig`），不是 GT 尺寸。当前 deblurring 配置 `scale=1` 所以 LQ=GT，但公式应通用
4. **插值方法**：spike crop 后 resize 仍用 `cv2.INTER_LINEAR`，与之前一致

## 5. 测试验证

- 验证 crop 后 spike patch 与 RGB patch 空间对齐：可视化叠加检查
- 验证各配置下 spike crop 区域不越界
- 验证现有非 rgbspike dataset 调用 `paired_random_crop` 不受影响（`_` 忽略第三返回值）
- 回归：现有测试套件通过

