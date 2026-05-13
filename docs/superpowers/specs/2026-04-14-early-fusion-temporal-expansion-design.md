# Early Fusion Temporal Expansion Design

**Date:** 2026-04-14  
**Status:** Draft  
**Scope:** EarlyFusionAdapter 重设计 — 空间对齐、时间展开、选帧输出、训练策略

> **Update (2026-04-16):** 本 spec 只涉及 restoration 主干的时间展开（N→N×S）。
> 光流路径（SCFlow）的时间轴对齐由子帧 encoding25 设计补充：
> `flow_spike` 也扩展到 `[B, N×S, 25, H, W]`，使 `get_flow_2frames()` 的时间维度自然匹配。
> 约束：`spike_flow.subframes` 必须等于 `spike_channels`。
> 详见 `docs/superpowers/specs/2026-04-16-scflow-subframe-encoding25-design.md`。

---

## 1. 背景与问题陈述

### 1.1 数据模态特性

| 模态 | 空间分辨率 | 时间分辨率 | 格式 |
|------|-----------|-----------|------|
| RGB | 高（如 1280×720） | 低（N 帧，如 6） | PNG，逐帧 |
| Spike | 低（如 640×360） | 高（每帧 S 个 TFP bin） | .dat → TFP 重建 |

**Spike 数据流**（在进入模型之前已完成重建）：

```
.dat 文件 (T_raw, spike_H, spike_W)  ← 原始二值 spike 流
  → SpikeStream.get_spike_matrix()
  → voxelize_spikes_tfp(num_channels=S)
  → (S, spike_H, spike_W) float32     ← 这是 EarlyFusionAdapter 的实际输入
```

`S` = `spike_channels`（TFP bins 数，默认 4，可配置）。  
Spike 在进入 fusion 时**已经是重建后的帧序列**，不是原始 .dat 文件。

### 1.2 现有 EarlyFusionAdapter 的问题

当前实现（`models/fusion/adapters/early.py`）存在两个未解决的问题：

**问题 1：空间分辨率不匹配**  
第 37 行断言 `rgb.shape[-2:] == spike.shape[-2:]`，但实际上 spike 空间分辨率低于 RGB，会直接报错。

**问题 2：残差连接 shape 冲突**  
VRT forward 在 fusion 之前提取 `x_lq_rgb = extract_rgb(x_lq)` 得到 `[B, N, 3, H, W]`，fusion 之后 `x` 变为 `[B, N*S, 3, H, W]`，末尾 `return x + x_lq_rgb` 因 shape 不匹配而失败。

### 1.3 设计目标

1. Early fusion 语义：**用 spike 增强 RGB，输出仍为 3 通道 RGB-like 张量**
2. 时间策略：**时间展开型**——将 N 帧展开为 N*S 帧，让 VRT backbone 在高时间分辨率上建模
3. 空间策略：将 spike 上采样至 RGB 分辨率（可学习）
4. 输出策略：支持 `restoration`（选 N 帧）和 `interpolation`（保留 N*S 帧）两种模式
5. 权重兼容：early fusion 输出 3 通道，VRT backbone `in_chans=3` 不变，pretrained 权重可直接加载
6. 训练策略：Stage A（冻结 VRT，只训练 fusion）→ Stage C（全参数 + 可选 LoRA）

---

## 2. 数据形状端到端

### 2.1 Dataset 输出（dual 模式）

```
L_rgb:   [B, N, 3, rgb_H, rgb_W]      # 模糊 RGB 帧
L_spike: [B, N, S, spike_H, spike_W]  # TFP 重建 spike，S=spike_channels
H:       [B, N, 3, rgb_H, rgb_W]      # 清晰 RGB GT
```

典型值：N=6, S=4, rgb_H×rgb_W=720×1280, spike_H×spike_W=360×640

### 2.2 EarlyFusionAdapter 内部

```
输入:
  rgb:   [B, N, 3, rgb_H, rgb_W]
  spike: [B, N, S, spike_H, spike_W]

Step 1 — SpikeUpsample:
  spike → [B, N, S, rgb_H, rgb_W]
  （双线性插值 + 2 层 refinement conv，见第 3.1 节）

Step 2 — 时间展开:
  rgb:   unsqueeze(2).expand(B, N, S, 3, H, W).reshape(B, N*S, 3, H, W)
  spike: reshape(B, N*S, 1, H, W)

Step 3 — Fusion Operator:
  operator(rgb_rep, spike_rep) → [B, N*S, 3, H, W]

输出: [B, N*S, 3, H, W]
```

### 2.3 VRT Backbone

```
输入:  [B, N*S, 3, H, W]           # in_chans=3，与 pretrained 权重兼容
conv_first 输入: [B, N*S, 3*9, H, W]  # pa_frames=2 → 3*9=27 channels
stages:  [B, N*S, embed_dim, H, W]
conv_last 输出: [B, N*S, 3, H, W]
```

VRT 的时间窗口注意力（`window_size[0]=6`）在 N*S 帧上运行，有效感知 spike 的高时间分辨率。

### 2.4 输出头

```
x_lq_rgb: [B, N, 3, H, W]  ← 在 fusion 之前提取，保存备用

restoration 模式:
  x = x[:, S//2::S, :, :, :]   # 选帧 → [B, N, 3, H, W]
  return x + x_lq_rgb           # 残差连接 ✅

interpolation 模式:
  x_lq_rgb_exp = x_lq_rgb.unsqueeze(2)
                              .expand(B, N, S, 3, H, W)
                              .reshape(B, N*S, 3, H, W)
  return x + x_lq_rgb_exp      # 残差连接 ✅
```

**选帧索引**：`S//2`（spike 窗口中心 bin，对应原始 RGB 帧时间戳）。  
选帧是纯索引操作，无参数，梯度只流向被选中的帧。

---

## 3. 模块设计

### 3.1 SpikeUpsample（新增子模块）

**位置**：`models/fusion/adapters/early.py`，作为 `EarlyFusionAdapter` 的子模块。

**设计**：双线性插值 + 轻量 refinement conv

```python
class SpikeUpsample(nn.Module):
    """将 spike 从低分辨率上采样至 RGB 分辨率。

    使用双线性插值处理任意 scale factor（spike 与 RGB 分辨率比不是整数倍），
    再用两层 Conv2d 学习修正稀疏 spike 数据的插值伪影。

    Args:
        spike_chans: spike 通道数（= S = spike_channels）
    """
    def __init__(self, spike_chans: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(spike_chans, spike_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spike_chans, spike_chans, kernel_size=3, padding=1),
        )

    def forward(self, spike: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        # spike: [B*N, S, spike_H, spike_W]
        x = F.interpolate(spike, size=(target_h, target_w),
                          mode='bilinear', align_corners=False)
        return self.refine(x)
```

**调用方式**（在 EarlyFusionAdapter.forward 中）：

```python
B, N, S, spike_H, spike_W = spike.shape
_, _, _, rgb_H, rgb_W = rgb.shape
if (spike_H, spike_W) != (rgb_H, rgb_W):
    spike_flat = spike.reshape(B * N, S, spike_H, spike_W)
    spike_flat = self.spike_upsample(spike_flat, rgb_H, rgb_W)
    spike = spike_flat.reshape(B, N, S, rgb_H, rgb_W)
```

**参数量**：2 × Conv2d(S, S, 3×3) ≈ 2 × S² × 9 参数。S=4 时约 288 个参数，极轻量。

**选择双线性而非 PixelShuffle 的原因**：spike 与 RGB 的分辨率比不是整数倍（如 640/1280=0.5 是整数，但 360/720=0.5 也是，实际数据集可能不同），双线性可处理任意比例。

### 3.2 EarlyFusionAdapter 重设计

**文件**：`models/fusion/adapters/early.py`

**完整 forward 逻辑**：

```python
def forward(self, rgb: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
    # rgb:   [B, N, 3, rgb_H, rgb_W]
    # spike: [B, N, S, spike_H, spike_W]

    if rgb.dim() != 5 or spike.dim() != 5:
        raise ValueError("rgb and spike must be 5D tensors [B, N, C, H, W]")

    B, N, rgb_chans, rgb_H, rgb_W = rgb.shape
    _, _, S, spike_H, spike_W = spike.shape

    # Step 1: 空间对齐（spike → RGB 分辨率）
    if (spike_H, spike_W) != (rgb_H, rgb_W):
        spike_flat = spike.reshape(B * N, S, spike_H, spike_W)
        spike_flat = self.spike_upsample(spike_flat, rgb_H, rgb_W)
        spike = spike_flat.reshape(B, N, S, rgb_H, rgb_W)

    # Step 2: 时间展开
    # rgb: [B, N, 3, H, W] → [B, N*S, 3, H, W]
    rgb_rep = rgb.unsqueeze(2).expand(B, N, S, rgb_chans, rgb_H, rgb_W)
    rgb_rep = rgb_rep.reshape(B, N * S, rgb_chans, rgb_H, rgb_W)
    # spike: [B, N, S, H, W] → [B, N*S, 1, H, W]
    spk = spike.reshape(B, N * S, 1, rgb_H, rgb_W)

    # Step 3: Fusion Operator（输出必须为 3 通道）
    return self.operator(rgb_rep, spk)
```

**Operator 输出约束**：所有 operator 的 `out_chans` 必须配置为 3，以保证 VRT backbone `in_chans=3` 与 pretrained 权重兼容。Factory 在构建时应验证此约束。

### 3.3 VRT Forward 修复

**文件**：`models/architectures/vrt/vrt.py`

**修改点 1**：`x_lq_rgb` 在 fusion 之前提取（现有逻辑已正确），保存 `[B, N, 3, H, W]`。

**修改点 2**：新增 `output_mode` 参数（从 `opt['netG']` 读取，默认 `'restoration'`）。

**修改点 3**：forward 末尾替换残差连接逻辑：

```python
# 原来（错误）:
# return x + x_lq_rgb  # shape 不匹配

# 修改后:
if self.output_mode == 'restoration':
    # S = spike_channels，从 config 读取
    S = self.spike_bins
    x = x[:, S // 2::S, :, :, :]   # [B, N*S, 3, H, W] → [B, N, 3, H, W]
    return x + x_lq_rgb             # [B, N, 3, H, W] ✅

elif self.output_mode == 'interpolation':
    B, NS, C, H, W = x.shape
    N = NS // self.spike_bins
    S = self.spike_bins
    x_lq_rgb_exp = (x_lq_rgb
                    .unsqueeze(2)
                    .expand(B, N, S, C, H, W)
                    .reshape(B, N * S, C, H, W))
    return x + x_lq_rgb_exp         # [B, N*S, 3, H, W] ✅
```

**修改点 4**：`__init__` 中新增配置解析：

```python
self.output_mode = opt.get('output_mode', 'restoration')
assert self.output_mode in {'restoration', 'interpolation'}
# spike_bins 从 fusion config 或 dataset config 读取
self.spike_bins = opt.get('spike', {}).get('reconstruction', {}).get('num_bins', 4)
```

---

## 4. Fusion Operator 适配

### 4.1 适用范围

空间上采样、时间展开、选帧逻辑全部在 **EarlyFusionAdapter** 层完成，operator 只需处理已展开的 `[B, N*S, 3, H, W]`（rgb）和 `[B, N*S, 1, H, W]`（spike）。

### 4.2 GatedFusionOperator

**兼容性**：完全兼容，无需修改。

**输入**（展开后）：
- `rgb_feat`: `[B, N*S, 3, H, W]`
- `spike_feat`: `[B, N*S, 1, H, W]`

**输出约束**：配置 `out_chans=3`。

**语义**：每个时间步（N*S 中的每一帧）独立做 gated fusion，spike 的 1 个 channel 对应该时间步的 TFP bin 值，gate 学习"这个时刻 spike 的可信度"。

### 4.3 MambaFusionOperator

**兼容性**：完全兼容，无需修改。

**输入**（展开后）：
- `rgb_feat`: `[B, N*S, 3, H, W]`
- `spike_feat`: `[B, N*S, 1, H, W]`

**序列建模**：Mamba 在 `steps=N*S` 的序列上运行（每个像素位置独立），序列长度从 N 增加到 N*S（如 6→24）。Mamba 是 O(N) 复杂度，无 OOM 风险。

**重要说明**：Mamba 是新引入的 operator，没有"N 帧时的语义"需要迁移。直接在 N*S 帧序列上建模，语义是"在高时间分辨率的 spike-enhanced 帧序列上做状态空间建模"，这正是我们想要的。

**输出约束**：配置 `out_chans=3`。

### 4.4 PaseFusionOperator（暂缓）

PASE 的 `in_chans=S`，它把所有 S 个 TFP bin 作为一个整体 `[B, S, H, W]` 处理（pixel-adaptive convolution 跨越整个时间维，联合建模 S 个 bin 的空间-时间关系）。

在时间展开设计里，spike 被 reshape 成 `[B, N*S, 1, H, W]`，每帧只有 1 个 spike channel。PASE 的 `in_chans` 就变成了 1，语义从"联合建模 S 个 bin"退化成"逐 bin 独立处理"，丧失了 PASE 的核心设计优势。

**结论**：PASE 需要单独设计适配方案（例如：在时间展开之前先用 PASE 提取特征，再展开），暂不纳入本次实施范围。

---

## 5. 训练策略

### 5.1 Stage A：冻结 VRT，只训练 EarlyFusionAdapter

**目标**：快速验证 early fusion 设计（SpikeUpsample + operator）是否有效。

**配置**：

```json
"training": {
    "freeze_backbone": true,
    "use_lora": false
}
```

**流程**：
1. 加载 pretrained VRT 权重（`in_chans=3`，全量加载，见第 6 节）
2. 冻结 VRT 所有参数（`requires_grad=False`）
3. 只训练 `EarlyFusionAdapter`（`SpikeUpsample` + operator）
4. Loss：选出的 N 帧 vs GT N 帧（Charbonnier loss，与原版 VRT 一致）
5. 优化器：Adam，lr=1e-4（fusion 模块参数量小，可用较大 lr）

**风险**：VRT 在 pretrain 时处理 N 帧，现在处理 N*S 帧，存在分布偏移。冻结状态下 VRT 无法适应，效果上限受限。但 Stage A 的目的是验证方向，不是追求最优性能。

**成功标准**：与 baseline（无 fusion，纯 RGB 输入）相比，PSNR/SSIM 有提升。

### 5.2 Stage C：全参数训练（可选 LoRA）

**目标**：让 VRT backbone 适应 N*S 帧的分布，充分发挥 early fusion 的效果。

**配置**：

```json
"training": {
    "freeze_backbone": false,
    "use_lora": true,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_target_modules": ["qkv", "proj"]
}
```

**LoRA 设计**：
- 目标层：VRT 每个 stage 的 attention QKV projection 和 output projection（`nn.Linear`）
- LoRA 参数：`W' = W + (alpha/rank) * B @ A`，其中 A、B 是低秩矩阵
- rank=8 时，参数量约为原 attention 参数的 1-5%
- Early fusion 模块全参数训练，VRT 只训练 LoRA adapter
- 实现：使用 `peft` 库或手动包装 linear 层

**Stage A 与 Stage C 是同一套代码**，只差 `training.freeze_backbone` 和 `training.use_lora` 两个 config 开关。

**训练顺序**：先跑 Stage A 验证方向 → 用 Stage A 的 fusion 权重初始化 Stage C → Stage C 全参数微调。

---

## 6. Pretrained 权重加载策略

### 6.1 DCN 类型与权重兼容性

原版 VRT 使用 DCNv2，S-VRT 当前 config 使用 DCNv4。两者权重形状不同，无法直接加载。

**解决方案**：partial weight loading——key 匹配则加载，不匹配则跳过（随机初始化）。

```python
def load_pretrained_partial(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)['params']
    model_state = model.state_dict()
    loaded, skipped = [], []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(model_state)
    return loaded, skipped
```

**Config 配置**：

```json
"netG": {
    "dcn": {
        "type": "dcnv2",        // 对齐原版 VRT → 全量加载
        // "type": "dcnv4",     // S-VRT 扩展 → DCN 层跳过，其余全量加载
        "apply_softmax": false
    }
}
```

### 6.2 对齐原版 VRT 的完整配置

当需要完全对齐原版 VRT（用于 baseline 对比或全量加载 pretrained 权重）：

```json
"netG": {
    "in_chans": 3,
    "fusion": { "enable": false },
    "spike_encoder": { "enable": false },
    "dcn": { "type": "dcnv2", "apply_softmax": false },
    "optical_flow": { "module": "spynet", "checkpoint": "..." }
}
```

光流模块（spynet/scflow/sea_raft）是独立权重，不影响 VRT backbone 的权重加载。

---

## 7. Config 新增字段

在 `options/gopro_rgbspike_local.json` 的 `netG` 节点下新增：

```json
"output_mode": "restoration",   // "restoration" | "interpolation"
"spike": {
    "reconstruction": {
        "type": "spikecv_tfp",
        "num_bins": 4            // = S = spike_channels，用于选帧索引
    }
},
"fusion": {
    "enable": true,
    "placement": "early",
    "operator": "gated",         // "gated" | "mamba"（PASE 暂缓）
    "out_chans": 3,              // 必须为 3
    "operator_params": {
        "hidden_chans": 32
    }
},
"training": {
    "freeze_backbone": true,     // Stage A: true；Stage C: false
    "use_lora": false,           // Stage C 可选: true
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_target_modules": ["qkv", "proj"]
}
```

---

## 8. 实施范围与边界

### 8.1 本次实施范围

| 组件 | 变更内容 |
|------|---------|
| `models/fusion/adapters/early.py` | 新增 `SpikeUpsample`；修复空间对齐；修复时间展开逻辑 |
| `models/architectures/vrt/vrt.py` | 修复残差连接 shape 冲突；新增 `output_mode`；新增 `spike_bins` 解析 |
| `models/fusion/factory.py` | 验证 `out_chans=3` 约束 |
| `options/gopro_rgbspike_local.json` | 新增 `output_mode`、`training` 节点；`dcn.type` 可切换 |
| `models/model_plain.py` 或训练脚本 | 新增 partial weight loading；新增 freeze_backbone 逻辑；新增 LoRA 包装 |

### 8.2 暂不实施

- PASE operator 的时间展开适配（需单独设计）
- LoRA 的具体实现（Stage C，后续）
- 插帧（interpolation）模式的数据集 GT 生成

### 8.3 不变的部分

- VRT backbone 的 stage 结构、attention、FFN 全部不变
- 光流模块（SCFlow/SpyNet）独立，不受影响
- Dataset 的 spike 重建流程不变（TFP 重建在 dataset 层完成）
- `input_pack_mode=dual` 的数据集接口不变

---

## 9. 关键设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| Spike 上采样方式 | 双线性 + 2 层 refinement conv | 处理任意 scale factor；conv 修正稀疏插值伪影；参数量极小 |
| 时间聚合方式 | 选帧（索引），不做 averaging | Averaging 会导致多帧 deblurred 图像叠加模糊；选帧是无损的 |
| 选帧索引 | `S//2`（中心 bin） | TFP 中心 bin 对应原始 RGB 帧时间戳 |
| Operator 输出通道 | 强制 `out_chans=3` | 保证 VRT backbone `in_chans=3`，pretrained 权重完全兼容 |
| 训练策略 | Stage A（冻结）→ Stage C（LoRA） | A 快速验证；C 充分利用 pretrained 权重同时控制训练成本 |
| DCN 兼容 | Partial loading，跳过 shape 不匹配的层 | DCNv2/v4 权重形状不同，无法强制加载 |
| PASE 处理 | 暂缓 | 时间展开后 spike 变为 1 channel/帧，PASE 的联合时间建模语义丧失，需单独设计 |
