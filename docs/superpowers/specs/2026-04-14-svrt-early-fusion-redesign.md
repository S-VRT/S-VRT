# S-VRT Early Fusion Redesign

**Date:** 2026-04-14  
**Status:** Draft  
**Scope:** EarlyFusionAdapter 重设计，含空间上采样、时间展开、VRT forward 修复、训练策略（Stage A / Stage C）、LoRA 配置、Pretrained 权重对齐、Operator 兼容性分析

---

## 1. 背景与动机

### 1.1 数据模态特性

| 模态 | 空间分辨率 | 时间分辨率 | 格式 |
|------|-----------|-----------|------|
| RGB | 高（如 1280×720） | 低（N 帧，如 6） | JPEG/PNG |
| Spike | 低（如 346×260） | 高（每帧 S 个 TFP bin） | `.dat` → TFP 重建 |

Spike 相机记录二值脉冲流（`.dat`），在数据集加载阶段由 `voxelize_spikes_tfp` 重建为 `(S, spike_H, spike_W)` float32 帧序列（TFP = Texture From Playback）。**Spike 进入 EarlyFusionAdapter 时已是 TFP 重建后的帧序列，而非原始 `.dat` 文件。**

`S`（TFP bins 数）= `spike_channels`，由 `spike.reconstruction.num_bins` 配置，默认 4。

### 1.2 当前 EarlyFusionAdapter 的两个问题

**问题 A：空间分辨率假设错误**

当前代码（`models/fusion/adapters/early.py` 第 37 行）断言 `rgb.shape[-2:] == spike.shape[-2:]`，但 RGB 和 Spike 的空间分辨率本来就不同，这在真实数据上会直接报错。

**问题 B：残差连接 shape 冲突**

VRT forward 在时间展开之前提取 `x_lq_rgb = extract_rgb(x_lq)`，shape 为 `[B, N, 3, H, W]`。early fusion 之后 `x` 变为 `[B, N*S, 3, H, W]`，末尾 `return x + x_lq_rgb` 因 temporal 维度不匹配而失败。

### 1.3 设计目标

1. 支持 RGB（高分辨率、低帧率）与 Spike TFP 帧（低分辨率、高帧率）的异分辨率融合
2. Early fusion 输出维持 3 通道（RGB-like），使 VRT backbone 的 `in_chans=3` 不变，pretrained 权重兼容
3. 时间展开（N → N*S 帧）由 adapter 完成，VRT backbone 在 N*S 帧上做时空建模
4. 支持 `output_mode`：`restoration`（选帧输出 N 帧）或 `interpolation`（输出 N*S 帧）
5. 支持 Stage A（冻结 VRT 只训练 fusion）和 Stage C（LoRA 微调 VRT + 全参数 fusion）
6. 支持通过 config 对齐原版 VRT 加载 pretrained 权重

---

## 2. 数据形状全链路

### 2.1 Dataset 输出（dual 模式）

```
L_rgb:   [B, N, 3,          rgb_H, rgb_W]   # 模糊 RGB，N 帧
L_spike: [B, N, S,        spike_H, spike_W]  # TFP 重建帧，N 帧 × S bins
H:       [B, N, 3,          rgb_H, rgb_W]   # 清晰 RGB GT，N 帧
```

其中 `N` = `num_frame`（如 6），`S` = `spike_channels`（如 4），`rgb_H` ≠ `spike_H`。

### 2.2 EarlyFusionAdapter 内部流程

```
输入:
  rgb:   [B, N, 3,   rgb_H, rgb_W]
  spike: [B, N, S, spike_H, spike_W]

Step 1 — SpikeUpsample（可学习）:
  spike → [B, N, S, rgb_H, rgb_W]
  （双线性插值 + 2 层 refinement conv）

Step 2 — 时间展开:
  rgb 沿 S 复制: [B, N, 3, H, W] → [B, N*S, 3, H, W]
  spike reshape: [B, N, S, H, W] → [B, N*S, 1, H, W]
  （其中 H=rgb_H, W=rgb_W）

Step 3 — Operator 融合:
  operator(rgb_rep=[B,N*S,3,H,W], spike_rep=[B,N*S,1,H,W])
  → [B, N*S, 3, H, W]   ← out_chans 固定为 3

输出: [B, N*S, 3, H, W]
```

### 2.3 VRT Backbone（输入 N*S 帧，in_chans=3 不变）

```
conv_first 输入: [B, N*S, 3*9, H, W]   （pa_frames=2 → 9 组对齐帧，每组 3 通道）
stages:          [B, N*S, embed_dim, H, W]  （时间窗口 attention 作用在 N*S 帧）
conv_last 输出:  [B, N*S, 3, H, W]
```

### 2.4 输出选帧（restoration 模式）

```
VRT 输出: [B, N*S, 3, H, W]

选帧（每组 S 帧取中心帧，对应原始 RGB 时间戳）:
  x_out = x[:, S//2 :: S, :, :, :]   # [B, N, 3, H, W]

残差连接（修复后）:
  return x_out + x_lq_rgb             # [B, N, 3, H, W] ✓
```

`S//2` = TFP 窗口中心 bin，语义上对应该 RGB 帧的时间戳。

### 2.5 输出（interpolation 模式）

```
x_lq_rgb_exp = x_lq_rgb.unsqueeze(2)
               .expand(B, N, S, 3, H, W)
               .reshape(B, N*S, 3, H, W)   # [B, N*S, 3, H, W]

return x + x_lq_rgb_exp                    # [B, N*S, 3, H, W] ✓
```

---

## 3. SpikeUpsample 模块设计

### 3.1 动机

Spike 的空间分辨率低于 RGB（scale factor 一般不是整数倍，如 1280/346 ≈ 3.7×），不能用 PixelShuffle。选择双线性插值 + 轻量可学习 refinement，原因：
- 双线性处理任意非整数 scale factor
- 2 层 conv 学习修正稀疏 spike 数据的插值伪影
- 参数量极少，不会抢 fusion operator 的学习份额

### 3.2 结构

```python
class SpikeUpsample(nn.Module):
    """将 Spike TFP 帧从低分辨率上采样到 RGB 分辨率。

    Args:
        spike_chans (int): TFP bins 数（即 S），作为通道维处理
    """
    def __init__(self, spike_chans: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(spike_chans, spike_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spike_chans, spike_chans, kernel_size=3, padding=1),
        )

    def forward(self, spike: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Args:
            spike: [B, N, S, spike_H, spike_W]
            target_h, target_w: RGB 的空间分辨率

        Returns:
            [B, N, S, target_h, target_w]
        """
        B, N, S, sH, sW = spike.shape
        x = spike.reshape(B * N, S, sH, sW)
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        x = self.refine(x)
        return x.reshape(B, N, S, target_h, target_w)
```

`SpikeUpsample` 是 `EarlyFusionAdapter` 的子模块，在 Step 1 调用。

---

## 4. EarlyFusionAdapter 重设计

### 4.1 完整 forward 逻辑

```python
def forward(self, rgb: Tensor, spike: Tensor) -> Tensor:
    """
    Args:
        rgb:   [B, N, 3,   rgb_H, rgb_W]
        spike: [B, N, S, spike_H, spike_W]

    Returns:
        [B, N*S, 3, rgb_H, rgb_W]   out_chans=3 固定
    """
    B, N, _, rgb_H, rgb_W = rgb.shape
    _, _, S, _, _ = spike.shape

    # Step 1: Spike 上采样到 RGB 分辨率
    spike = self.spike_upsample(spike, rgb_H, rgb_W)  # [B, N, S, rgb_H, rgb_W]

    # Step 2: 时间展开
    rgb_rep = rgb.unsqueeze(2).expand(B, N, S, 3, rgb_H, rgb_W)
    rgb_rep = rgb_rep.reshape(B, N * S, 3, rgb_H, rgb_W)     # [B, N*S, 3, H, W]
    spk_rep = spike.reshape(B, N * S, 1, rgb_H, rgb_W)        # [B, N*S, 1, H, W]

    # Step 3: Operator 融合
    return self.operator(rgb_rep, spk_rep)                     # [B, N*S, 3, H, W]
```

### 4.2 Operator 约束

所有 operator 在用于 EarlyFusionAdapter 时，必须满足：
- `out_chans = 3`：输出 RGB-like 特征，保持 VRT backbone `in_chans=3`
- `spike_chans = 1`：时间展开后每个 spike 帧只有 1 个 bin

---

## 5. VRT Forward 修复

### 5.1 修改点一：x_lq_rgb 保存时机

在 early fusion 之前（即 `x` 仍为 N 帧时）提取并保存：

```python
x_lq_rgb = self.extract_rgb(x_lq)    # [B, N, 3, H, W]，在 fusion 前
```

现有代码已是此顺序，无需改动。

### 5.2 修改点二：记录 S（spike bins 数）

在 early fusion 调用时记录 `S`，供末尾选帧使用：

```python
if self.fusion_enabled and fusion_placement in {'early', 'hybrid'}:
    rgb   = x[:, :, :3,  :, :]
    spike = x[:, :, 3:,  :, :]
    S = spike.shape[2]            # TFP bins 数
    x = self.fusion_adapter(rgb=rgb, spike=spike)   # [B, N*S, 3, H, W]
else:
    S = 1
```

### 5.3 修改点三：output_mode 分支（末尾）

```python
output_mode = self.opt.get('output_mode', 'restoration')   # 默认 restoration

if output_mode == 'restoration':
    # 每组 S 帧取中心帧（对应原始 RGB 时间戳）
    x = x[:, S // 2 :: S, :, :, :]        # [B, N, 3, H, W]
    return x + x_lq_rgb                    # [B, N, 3, H, W] ✓

elif output_mode == 'interpolation':
    B, N_orig = x_lq_rgb.shape[:2]
    x_lq_rgb_exp = (
        x_lq_rgb.unsqueeze(2)
        .expand(B, N_orig, S, 3, x_lq_rgb.shape[-2], x_lq_rgb.shape[-1])
        .reshape(B, N_orig * S, 3, x_lq_rgb.shape[-2], x_lq_rgb.shape[-1])
    )
    return x + x_lq_rgb_exp                # [B, N*S, 3, H, W] ✓
```

`output_mode` 通过 `netG.output_mode` 配置项控制，默认 `'restoration'`。

---

## 6. 训练策略

### 6.1 Stage A：冻结 VRT，只训练 EarlyFusionAdapter

**目标：** 快速验证 early fusion 设计（SpikeUpsample + operator）是否有效，成本极低。

**流程：**
1. 加载 pretrained VRT 权重（`in_chans=3`，完全兼容）
2. 冻结 VRT backbone 所有参数（`requires_grad=False`）
3. 只有 `EarlyFusionAdapter`（`SpikeUpsample` + operator）的参数参与训练
4. Loss：`output_mode='restoration'` 下选出的 N 帧 vs GT N 帧，Charbonnier Loss

**已知局限：** VRT 的 pretrained 权重是在纯 N 帧 RGB 上训练的，现在喂 N*S 帧，存在分布偏移。冻结状态下 VRT 无法自适应。Stage A 的主要价值是验证 fusion 方向，不期望达到 Stage C 的上限。

**Config 关键字段：**
```json
{
  "netG": {
    "in_chans": 3,
    "output_mode": "restoration",
    "fusion": {
      "enable": true,
      "placement": "early",
      "out_chans": 3
    }
  },
  "training": {
    "freeze_backbone": true,
    "use_lora": false
  }
}
```

### 6.2 Stage C：LoRA 微调 VRT + 全参数 EarlyFusionAdapter

**目标：** 让 VRT backbone 适应 N*S 帧的输入分布，同时控制训练成本。

**关键设计：Stage A 和 Stage C 使用完全相同的代码路径，只差 config 开关。**

**流程：**
1. 加载 Stage A 训练好的 EarlyFusionAdapter 权重（可选）
2. 加载 pretrained VRT 权重初始化 backbone
3. 对 VRT 所有 attention 层的 QKV Linear 注入 LoRA adapter（用 `peft` 库或手动实现）
4. EarlyFusionAdapter 全参数训练，VRT 只有 LoRA 参数参与训练，其余冻结
5. Loss：同 Stage A

**LoRA 配置（`netG.lora`）：**
```json
{
  "lora": {
    "enable": true,
    "rank": 8,
    "alpha": 16,
    "target_modules": ["qkv", "proj"]
  }
}
```

**Config 关键字段（Stage C）：**
```json
{
  "training": {
    "freeze_backbone": false,
    "use_lora": true
  }
}
```

---

## 7. Pretrained 权重加载策略

### 7.1 问题根源

原版 VRT 的 DCN 使用 DCNv2，当前 S-VRT 默认配置使用 DCNv4。DCNv2 与 DCNv4 的权重张量形状不同，无法直接全量加载。

### 7.2 解决方案：Partial Loading + dcn.type 配置

在权重加载时，做 key-by-key 匹配：
- **shape 匹配**：正常加载
- **shape 不匹配**（DCN 层）：跳过，保留随机初始化（或 DCNv4 专用初始化）
- **key 不存在**：跳过

| 场景 | `dcn.type` | `fusion.enable` | `in_chans` | Pretrained 加载结果 |
|------|-----------|-----------------|------------|---------------------|
| 完全对齐原版 VRT | `dcnv2` | `false` | 3 | 全量加载，无跳过 |
| S-VRT 正常训练 | `dcnv4` | `true` | 3（fusion 后） | DCN 层跳过，其余全量加载 |

**光流模块独立权重**（spynet / scflow / sea_raft）：通过 `optical_flow.checkpoint` 单独加载，与 VRT backbone pretrained 权重互不干扰。

### 7.3 配置示例（对齐 VRT）

```json
{
  "netG": {
    "in_chans": 3,
    "dcn": { "type": "dcnv2", "apply_softmax": false },
    "fusion": { "enable": false },
    "spike_encoder": { "enable": false },
    "output_mode": "restoration"
  }
}
```

---

## 8. Operator 兼容性分析

在 EarlyFusionAdapter 的时间展开设计下，spike 输入从 `[B, N, S, H, W]` 变为 `[B, N*S, 1, H, W]`（每帧 1 个 spike channel）。

### 8.1 Gated（✅ 完全兼容，当前阶段实施）

- `spike_chans=1`，输出 `rgb_proj(rgb) + gate * fused`，语义：spike 以门控方式增强 RGB
- 无特殊约束

### 8.2 Mamba（✅ 完全兼容，当前阶段实施）

- Mamba 作为新引入的 operator，没有"N 帧时的语义"需要迁移，直接在 N*S 帧序列上建模
- 序列长度从 N（如 6）变为 N*S（如 24），Mamba 是 O(N) 复杂度，无 OOM 风险
- `spike_chans=1` 完全兼容

### 8.3 PASE（⚠️ 暂缓，需单独讨论）

PASE 的设计假设与时间展开存在语义冲突：
- PASE（`PixelAdaptiveSpikeEncoder`）接收 `[B, L, H, W]`，将 L 个 bin **联合**做 pixel-adaptive convolution，输出 `[B, out_chans, H, W]`
- 时间展开后，spike 变为 `[B, N*S, 1, H, W]`（每帧 1 bin），PASE 的 `in_chans` 退化为 1，无法利用跨 bin 的时间相关性，其核心设计优势完全失效
- PASE 的合理输入应是完整的 S 个 bin（`[B, S, H, W]`），而非时间展开后的单 bin

**后续讨论方向（不在本次实施范围内）：**
- 在时间展开前，为每个 RGB 帧单独调用 PASE（输入 `[B*N, S, H, W]`，输出 `[B*N, out_chans, H, W]`），再展开/复制以对齐时间维
- 这意味着 PASE 可能更适合一种"先提取 spike feature，再融合"的 early-early fusion 路径

---

## 9. 需要修改的文件清单

| 文件 | 修改内容 |
|------|---------|
| `models/fusion/adapters/early.py` | 新增 `SpikeUpsample` 子模块；修改 forward 加入上采样 + 时间展开逻辑；移除空间分辨率相等断言 |
| `models/architectures/vrt/vrt.py` | forward 末尾：新增 `output_mode` 分支（选帧 vs 展开残差）；fusion 调用时记录 S；修复残差 shape 冲突 |
| `models/architectures/vrt/vrt.py` | `__init__`：新增 `dcn.type=dcnv2` 分支；新增 LoRA 注入逻辑（当 `training.use_lora=true` 时） |
| `models/model_plain.py` | 支持 `freeze_backbone` 配置项；partial weight loading（shape mismatch 时跳过） |
| `options/gopro_rgbspike_local.json` | 新增 `output_mode`、`training.freeze_backbone`、`training.use_lora`、`lora` 配置块 |
| `options/gopro_rgbspike_server.json` | 同上 |
| `options/vrt_aligned.json`（新文件） | 对齐原版 VRT 的配置文件，`dcn.type=dcnv2`，`fusion.enable=false` |

---

## 10. 不在本次范围内的内容

- **PASE operator 在时间展开下的适配**：语义冲突较大，需单独设计
- **数据集侧的 output_mode 改造**：`interpolation` 模式需要 N*S 帧的 GT（需要插帧网络生成），当前数据集只有 N 帧 GT
- **Stage C LoRA 的具体训练脚本**：本次记录设计，实施在 Stage A 验证之后
- **Spike 空间对齐的高级方案**（如 Deformable Upsample）：双线性 + refinement 是当前选择，后续可替换

---

## 11. 关键设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| Spike 上采样方式 | 双线性 + 2 层 refinement conv | 处理任意 scale factor；参数量少不抢 fusion 学习份额 |
| Operator 输出通道数 | 固定 3 | 保持 VRT `in_chans=3`，pretrained 权重完全兼容 |
| 时间聚合方式 | 选帧（`x[:, S//2::S]`） | 纯索引，无参数，无模糊/拖影；中心 bin 对应 RGB 时间戳 |
| Stage A vs C 架构 | 同一套代码 + config 开关 | A 和 C 只差 `freeze_backbone` 和 `use_lora`，无需维护两套代码 |
| DCN 权重冲突 | Partial loading（shape 不匹配跳过） | 允许在 DCNv4 配置下仍加载 VRT backbone 的其他层权重 |
| PASE 处理 | 暂缓 | 时间展开后 PASE 退化为 1-bin 处理，其联合时间建模能力完全失效，需单独设计 |
